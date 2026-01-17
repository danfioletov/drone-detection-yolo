import bpy
import os
import math
import random
import argparse
from mathutils import Vector

# ----------------------------
# Helpers
# ----------------------------

def get_meshes_in_scene():
    """Return all mesh objects currently present in the opened .blend scene."""
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        raise RuntimeError("No MESH objects found in the opened .blend scene.")
    return meshes

def set_origin_to_geometry(obj):
    """Set object origin to its geometry bounds center (improves rotation behavior)."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.select_set(False)

def compute_scene_center(objects):
    """Compute approximate world-space center and diagonal length of combined bounding boxes."""
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))

    for obj in objects:
        for v in obj.bound_box:
            vw = obj.matrix_world @ Vector(v)
            mins = Vector((min(mins.x, vw.x), min(mins.y, vw.y), min(mins.z, vw.z)))
            maxs = Vector((max(maxs.x, vw.x), max(maxs.y, vw.y), max(maxs.z, vw.z)))

    center = (mins + maxs) / 2.0
    diag = (maxs - mins).length
    return center, diag

def ensure_camera():
    """Create and activate a camera."""
    cam_data = bpy.data.cameras.new(name="DatasetCamera")
    cam_data.lens = 35
    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000

    cam_obj = bpy.data.objects.new("DatasetCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj

def ensure_light():
    """Create a sun light."""
    light_data = bpy.data.lights.new(name="DatasetSun", type='SUN')
    light_obj = bpy.data.objects.new("DatasetSun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    return light_obj

def look_at(obj, target: Vector):
    """Rotate object to look at target point."""
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def setup_render(output_dir, res=640):
    """Configure render settings for transparent RGBA PNG."""
    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100

    # transparent background
    scene.render.film_transparent = True

    # PNG RGBA
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.compression = 15

    os.makedirs(output_dir, exist_ok=True)

def random_camera_pose(
    cam,
    target,
    base_radius,
    min_factor=0.5,   # <- не підходити ближче 1.3*base_radius
    max_factor=3.0,   # <- не відлітати занадто далеко
    el_min_deg=5,
    el_max_deg=90
):
    """Place camera on a random point on a sphere around target with constrained distance."""
    az = random.uniform(0, 2 * math.pi)
    el = random.uniform(math.radians(el_min_deg), math.radians(el_max_deg))

    r_min = base_radius * min_factor
    r_max = base_radius * max_factor
    r = random.uniform(r_min, r_max)

    x = target.x + r * math.cos(el) * math.cos(az)
    y = target.y + r * math.cos(el) * math.sin(az)
    z = target.z + r * math.sin(el)

    cam.location = Vector((x, y, z))
    look_at(cam, target)

def random_object_rotation(objects):
    """Randomly rotate all mesh objects (keeps relative structure if multiple parts)."""
    rx = random.uniform(0, 2 * math.pi)
    ry = random.uniform(0, 2 * math.pi)
    rz = random.uniform(0, 2 * math.pi)

    for obj in objects:
        obj.rotation_euler = (rx, ry, rz)

def random_light(light_obj):
    """Randomize sun direction and energy."""
    light_obj.rotation_euler = (
        random.uniform(-math.pi, math.pi),
        random.uniform(-math.pi, math.pi),
        random.uniform(-math.pi, math.pi),
    )
    light_obj.data.energy = random.uniform(2.0, 10.0)

# ----------------------------
# Main
# ----------------------------

def main(output_dir, num=300, res=640):
    setup_render(output_dir, res=res)

    meshes = get_meshes_in_scene()

    # Normalize origins (optional but recommended)
    for obj in meshes:
        set_origin_to_geometry(obj)

    # Move model to origin (so camera target is stable)
    center, diag = compute_scene_center(meshes)
    for obj in meshes:
        obj.location -= center

    target = Vector((0, 0, 0))
    _, diag = compute_scene_center(meshes)

    cam = ensure_camera()
    light = ensure_light()

    # distance heuristic
    base_radius = max(2.0, diag * 1.5)

    for i in range(num):
        random_object_rotation(meshes)
        random_camera_pose(cam, target, base_radius=base_radius, 
                           min_factor=0.5, max_factor=3.0, el_min_deg=5, el_max_deg=90)
        random_light(light)

        bpy.context.scene.render.filepath = os.path.join(output_dir, f"sprite_{i:05d}.png")
        bpy.ops.render.render(write_still=True)

        if (i + 1) % 25 == 0:
            print(f"Rendered {i + 1}/{num}")

    print(f"Done. Rendered {num} sprites to: {output_dir}")

if __name__ == "__main__":
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output folder for sprites")
    parser.add_argument("--num", type=int, default=300, help="Number of sprites")
    parser.add_argument("--res", type=int, default=640, help="Resolution (square)")
    args = parser.parse_args(argv)

    main(args.out, num=args.num, res=args.res)