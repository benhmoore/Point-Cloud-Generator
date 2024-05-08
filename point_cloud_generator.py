bl_info = {
    "name": "Point Cloud Generator",
    "blender": (4, 0, 0),
    "category": "Object",
    "description": "Generates a point cloud and reference camera positions from the selected meshes, sampling colors from rendered frames, for use in NeRF or 3D-Gaussian reconstruction.",
}

import bpy
import random
import bmesh
import json
import os
import mathutils
from bpy.props import IntProperty, StringProperty, PointerProperty, BoolProperty
from bpy.types import Operator, Panel, PropertyGroup


def ensure_directory_exists(path):
    """Ensure the directory exists, create if it does not."""
    if not os.path.exists(path):
        os.makedirs(path)


def sample_points(mesh_obj, resolution):
    """Sample points on the mesh faces based on the face area."""

    # Triangulate the mesh
    mesh = bmesh.new()
    mesh.from_mesh(mesh_obj.data)
    bmesh.ops.triangulate(mesh, faces=mesh.faces)

    # Sample points on the mesh faces, based on the face area
    points = []
    for face in mesh.faces:
        face_area = face.calc_area()
        num_samples = max(1, int(resolution * face_area))

        for _ in range(num_samples):
            u = random.random()
            v = random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            w = 1 - u - v

            # Use stratified sampling to distribute points evenly on the face
            point = face.verts[0].co * u + face.verts[1].co * v + face.verts[2].co * w
            points.append((point[0], point[1], point[2]))

    mesh.free()
    return points


def export_points_as_ply_manual(points, colors, file_path):
    """Export points and colors to a PLY file."""

    with open(file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point, color in zip(points, colors):
            # Swapping y and z, and inverting the new z (original y) for Y-up orientation
            f.write(
                f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n"
            )


# def create_point_cloud(points):
#     """Create a point cloud object from the given points."""

#     # Create a new mesh data block to store the points
#     mesh_data = bpy.data.meshes.new("point_cloud_mesh")

#     # Create a new object with the mesh data
#     point_cloud = bpy.data.objects.new("Point Cloud", mesh_data)
#     bpy.context.collection.objects.link(point_cloud)
#     bpy.context.view_layer.objects.active = point_cloud
#     point_cloud.select_set(True)

#     # Create a new bmesh to add the points
#     mesh = bmesh.new()
#     for point in points:
#         mesh.verts.new(point)
#     mesh.to_mesh(mesh_data)
#     mesh.free()

#     return point_cloud


def create_cameras_around_object(
    obj, subdivisions=2, radius=10, focal_length=50, exclude_bottom_half=False
):
    """Create cameras around the object using an icosphere as a template."""

    # Create an icosphere to place the cameras
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions, radius=radius, location=obj.location
    )
    ico_sphere = bpy.context.object

    # Create cameras at each vertex of the icosphere
    cameras = []
    for vertex in ico_sphere.data.vertices:
        if exclude_bottom_half and vertex.co.z < 0:
            continue

        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        camera.location = ico_sphere.matrix_world @ vertex.co

        direction = -(obj.location - camera.location)
        camera.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        camera.data.lens = focal_length

        bpy.context.scene.collection.objects.link(camera)
        cameras.append(camera)

    bpy.data.objects.remove(ico_sphere, do_unlink=True)

    if cameras:
        bpy.context.scene.camera = cameras[
            0
        ]  # Set the first camera as the active camera

    return cameras


def project_point_to_frame(point, frame):
    """Project a 3D point to a 2D point on the camera frame."""

    # Extract camera parameters from the frame data
    w = frame["w"]
    h = frame["h"]
    fl_x = frame["fl_x"]
    fl_y = frame["fl_y"]
    cx = frame["cx"]
    cy = frame["cy"]
    transform_matrix = frame["transform_matrix"]

    # Convert the transform matrix to a Blender matrix
    matrix = mathutils.Matrix(transform_matrix)

    # Convert the 3D point to a Blender vector
    point_vec = mathutils.Vector(point)

    # Transform the point using the camera's transform matrix
    transformed_point = matrix @ point_vec

    # Project the transformed point onto the camera's view plane
    projected_x = (transformed_point.x * fl_x / transformed_point.z) + cx
    projected_y = (transformed_point.y * fl_y / transformed_point.z) + cy

    return (projected_x, projected_y)


def is_within_frame(projected_pos, frame):
    """Check if the projected position is within the frame boundaries."""
    w = frame["w"]
    h = frame["h"]
    return 0 <= projected_pos[0] < w and 0 <= projected_pos[1] < h


def sample_color(projected_pos, frame):
    """Sample the color value at the projected position in the rendered frame."""

    # Load the rendered frame image
    image = bpy.data.images.load(frame["file_path"])

    # Calculate the pixel coordinates within the image
    x = int(projected_pos[0])
    y = int(projected_pos[1])

    # Sample the color value at the pixel coordinates
    color = image.pixels[
        y * image.size[0] * 4 + x * 4 : y * image.size[0] * 4 + x * 4 + 3
    ]

    return [int(c * 255) for c in color]


class OBJECT_OT_generate_point_cloud(Operator):
    bl_idname = "object.generate_point_cloud"
    bl_label = "Generate Point Cloud"
    bl_description = (
        "Generate a point cloud from the selected meshes and camera positions"
    )
    bl_options = {"PRESET"}

    points = []  # Store the generated points

    def execute(self, context):
        """Generate a point cloud from the selected meshes and camera positions."""
        settings = context.scene.point_cloud_settings
        selected_objects = context.selected_objects

        self.points.clear()  # Clear the points list before generating new points

        # Spawn points for each selected mesh object
        for obj in selected_objects:
            if obj.type == "MESH":
                points = sample_points(obj, settings.resolution)
                self.points.extend(points)

        if not self.points:
            self.report({"WARNING"}, "No mesh objects selected")
            return {"CANCELLED"}

        # Ensure the output directory exists
        output_path = bpy.path.abspath(settings.output_path)
        ensure_directory_exists(output_path)

        # Setup cameras
        active_object = context.active_object
        if active_object:
            create_cameras_around_object(
                active_object,
                subdivisions=settings.subdivisions,
                radius=settings.camera_radius,
                focal_length=settings.focal_length,
                exclude_bottom_half=settings.exclude_bottom_half,
            )
        else:
            self.report({"WARNING"}, "No active object found for camera placement")

        return {"FINISHED"}


class OBJECT_OT_render_scene(Operator):
    bl_idname = "object.render_scene"
    bl_label = "Render Scene"
    bl_description = (
        "Render from each camera, sample point colors, and export point cloud"
    )

    @classmethod
    def poll(cls, context):
        return any(obj.type == "CAMERA" for obj in context.scene.objects)

    def execute(self, context):
        """Render the scene from each camera, sample point colors, and export point cloud."""

        addon_prefs = context.preferences.addons[__name__].preferences
        scn = context.scene
        original_camera = scn.camera

        # Ensure the output and image directories exists
        output_path = bpy.path.abspath(scn.point_cloud_settings.output_path)
        ensure_directory_exists(output_path)

        images_path = os.path.join(output_path, addon_prefs.images_folder_name)
        ensure_directory_exists(images_path)

        camera_details = {
            "camera_model": "OPENCV",
            "frames": [],
            "ply_file_path": addon_prefs.ply_file_name,
        }

        # Set the is_rendering flag to True to indicate that rendering is in progress
        # This feature is not fully implemented yet
        scn.point_cloud_settings.is_rendering = True

        for obj in scn.objects:
            if obj.type == "CAMERA":
                if not scn.point_cloud_settings.is_rendering:
                    break  # Interrupt the rendering if is_rendering is set to False

                # Set the current camera as the active camera (for viewport visualization of progress)
                bpy.context.scene.camera = obj

                scn.camera = obj
                image_filename = f"{obj.name}.png"
                scn.render.filepath = os.path.join(images_path, image_filename)
                bpy.ops.render.render(write_still=True)
                print(f"Rendered from camera: {obj.name}")

                # Get render resolution
                render_resolution_x = scn.render.resolution_x
                render_resolution_y = scn.render.resolution_y

                # Calculate intrinsic parameters
                sensor_width_mm = obj.data.sensor_width
                focal_length_mm = obj.data.lens
                pixel_aspect_ratio = (
                    scn.render.pixel_aspect_x / scn.render.pixel_aspect_y
                )

                if obj.data.sensor_fit == "VERTICAL":
                    # Vertical fit, calculate based on sensor height
                    sensor_height_mm = sensor_width_mm / pixel_aspect_ratio
                    fl_y = (focal_length_mm / sensor_height_mm) * render_resolution_y
                    fl_x = fl_y / pixel_aspect_ratio
                else:
                    # Horizontal fit or Auto fit, calculate based on sensor width
                    fl_x = (focal_length_mm / sensor_width_mm) * render_resolution_x
                    fl_y = fl_x * pixel_aspect_ratio

                cx = render_resolution_x / 2
                cy = render_resolution_y / 2

                # Frame details for camera, mimicking NerfStudio's format
                frame_details = {
                    "file_path": scn.render.filepath,
                    "w": render_resolution_x,
                    "h": render_resolution_y,
                    "fl_x": fl_x,
                    "fl_y": fl_y,
                    "cx": cx,
                    "cy": cy,
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                    "k3": 0.0,
                    "transform_matrix": [list(row) for row in obj.matrix_world],
                }

                camera_details["frames"].append(frame_details)

        scn.camera = original_camera
        scn.point_cloud_settings.is_rendering = False

        # Write camera details to JSON
        json_path = os.path.join(output_path, addon_prefs.transforms_file_name)
        with open(json_path, "w") as f:
            json.dump(camera_details, f, indent=4)

        # Sample point colors from rendered frames
        points = OBJECT_OT_generate_point_cloud.points

        # For every point, set the color to red by default
        colors = []

        # Sample colors in batches to (hopefully) reduce memory pressure
        batch_size = 1000  # TODO: Expose this as a preferences option
        for batch_start in range(0, len(points), batch_size):
            batch_end = min(batch_start + batch_size, len(points))
            batch_points = points[batch_start:batch_end]
            batch_colors = [[0, 0, 0] for _ in range(len(batch_points))]

            for i, point in enumerate(batch_points):
                visible_frames = []
                for frame in camera_details["frames"]:
                    projected_pos = project_point_to_frame(point, frame)

                    # If the projected position is within the frame boundaries
                    if is_within_frame(projected_pos, frame):

                        # Convert the transform matrix back to a Blender Matrix object
                        transform_matrix = mathutils.Matrix(frame["transform_matrix"])

                        # Perform a visibility check using ray casting
                        camera_pos = transform_matrix.translation
                        direction = mathutils.Vector(point) - camera_pos
                        ray_cast_result = bpy.context.scene.ray_cast(
                            bpy.context.view_layer.depsgraph, camera_pos, direction
                        )

                        # If the ray cast intersects a surface near the given point,
                        # consider the point visible from the camera
                        if (
                            ray_cast_result[0]
                            and (ray_cast_result[1] - mathutils.Vector(point)).length
                            < 0.001
                        ):
                            visible_frames.append((frame, projected_pos))
                            break  # Break after the first visible frame
                            # TODO: Implement averaging of colors from multiple frames

                # If the point is visible from at least one camera
                if visible_frames:
                    frame, projected_pos = visible_frames[
                        0
                    ]  # Use the first visible frame for color sampling (for now)
                    batch_colors[i] = sample_color(projected_pos, frame)

                print(
                    f"Sampling color for point: {point}, Color: {batch_colors[i]}"
                )  # Print statement for debugging

            # Extend the colors list with the batch colors
            colors.extend(batch_colors)

            del batch_points
            del batch_colors
            visible_frames.clear()

        # Export point cloud with colors
        file_path = os.path.join(output_path, addon_prefs.ply_file_name)
        export_points_as_ply_manual(points, colors, file_path)

        return {"FINISHED"}


class OBJECT_OT_cancel_render(Operator):
    bl_idname = "object.cancel_render"
    bl_label = "Cancel Render"
    bl_description = "Cancel the rendering process"

    def execute(self, context):
        context.scene.point_cloud_settings.is_rendering = False
        return {"FINISHED"}


class PointCloudSettings(PropertyGroup):
    resolution: IntProperty(
        name="Resolution",
        description="Controls the number of points per unit area",
        default=1000,
        min=1,
        max=10000,
    )
    subdivisions: IntProperty(
        name="Icosphere Subdivisions",
        description="Subdivisions of the icosphere for camera placement",
        default=2,
        min=1,
        max=5,
    )
    camera_radius: IntProperty(
        name="Camera Sphere Radius",
        description="Radius of the sphere on which cameras are placed",
        default=10,
        min=1,
        max=50,
    )
    focal_length: IntProperty(
        name="Camera Focal Length",
        description="Focal length of the cameras (in mm)",
        default=50,
        min=1,
        max=300,
    )
    exclude_bottom_half: BoolProperty(
        name="Exclude Bottom Half",
        description="Exclude cameras on the bottom half of the icosphere",
        default=False,
    )

    output_path: StringProperty(
        name="Output Path",
        description="Directory to save all outputs",
        default="//outputs",
        subtype="DIR_PATH",
    )
    is_rendering: BoolProperty(
        name="Is Rendering",
        description="Indicates if the rendering is in progress",
        default=False,
    )


class VIEW3D_PT_point_cloud_exporter(Panel):
    bl_label = "Point Cloud Generator"
    bl_category = "Tool"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.point_cloud_settings

        # General Settings
        general_box = layout.box()
        general_box.label(text="General Settings")
        general_box.prop(settings, "output_path", text="Output Directory")

        # Point Cloud Settings Section
        point_cloud_box = layout.box()
        point_cloud_box.label(text="Point Cloud Settings")
        point_cloud_box.prop(settings, "resolution")
        point_cloud_box.operator(
            OBJECT_OT_generate_point_cloud.bl_idname,
            text="Generate Point Cloud and Camera Positions",
        )

        # Camera Setup Section
        camera_box = layout.box()
        camera_box.label(text="Camera Setup")
        camera_box.prop(settings, "subdivisions", text="Icosphere Subdivisions")
        camera_box.prop(settings, "camera_radius", text="Camera Sphere Radius")
        camera_box.prop(settings, "focal_length", text="Camera Focal Length")
        camera_box.prop(settings, "exclude_bottom_half", text="Exclude Bottom Half")

        # Rendering Options Section
        render_box = layout.box()
        render_box.label(text="Rendering Options")
        render_box.operator(
            OBJECT_OT_render_scene.bl_idname, text="Render Scene and Export Point Cloud"
        )

        if settings.is_rendering:
            render_box.operator(OBJECT_OT_cancel_render.bl_idname, text="Cancel Render")


class POINTCLOUD_PT_preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    ply_file_name: StringProperty(
        name="PLY File Name",
        description="Name of the PLY file",
        default="reconstruction.ply",
    )

    transforms_file_name: StringProperty(
        name="Transforms File Name",
        description="Name of the transforms JSON file",
        default="transforms.json",
    )

    images_folder_name: StringProperty(
        name="Images Folder Name",
        description="Name of the folder to store rendered images",
        default="images",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "ply_file_name")
        layout.prop(self, "transforms_file_name")
        layout.prop(self, "images_folder_name")


def register():
    bpy.utils.register_class(OBJECT_OT_generate_point_cloud)
    bpy.utils.register_class(OBJECT_OT_render_scene)
    bpy.utils.register_class(OBJECT_OT_cancel_render)
    bpy.utils.register_class(VIEW3D_PT_point_cloud_exporter)
    bpy.utils.register_class(PointCloudSettings)
    bpy.utils.register_class(POINTCLOUD_PT_preferences)
    bpy.types.Scene.point_cloud_settings = PointerProperty(type=PointCloudSettings)


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_generate_point_cloud)
    bpy.utils.unregister_class(OBJECT_OT_render_scene)
    bpy.utils.unregister_class(OBJECT_OT_cancel_render)
    bpy.utils.unregister_class(VIEW3D_PT_point_cloud_exporter)
    bpy.utils.unregister_class(PointCloudSettings)
    bpy.utils.unregister_class(POINTCLOUD_PT_preferences)
    del bpy.types.Scene.point_cloud_settings


if __name__ == "__main__":
    register()
