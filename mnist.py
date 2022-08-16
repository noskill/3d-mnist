import bpy
import sys
import argparse

# load other scripts from current blend file
#for name, text in bpy.data.texts.items():
#    if name.endswith('.py') and name[:-3] not in sys.modules:
#        sys.modules[name[:-3]] = text.as_module()
import os
import bmesh
import imp
import time
import math
import random
import sys
import json
from mathutils import Vector

import mnist_util

imp.reload(mnist_util)



def parse_args(args):
    parser = argparse.ArgumentParser(description='render mnist in 3d')
    parser.add_argument('--source',
                    help='source directory with mnist')
    parser.add_argument('--render',
                    help='render output directory')
    parser.add_argument('--meta',
                    help='metadata output directory')
    parser.add_argument('--blend',
                    help='blend file to use')
    return parser.parse_args(args)


def main(source_path, meta_path, render_path, num=0):
    bpy.ops.object.select_all(action='DESELECT')

    mnist_filename = random.choice(os.listdir(source_path))
    path = os.path.join(source_path, mnist_filename)

    print(path)

    img = bpy.data.images.load(path)
    print(img.depth)

    make_obj(img)
    randomize_camera()

    label = mnist_filename.split('_')[2].split('.')[0]
    out_path = '0' * (6 - len(str(num))) + str(num) + '_' + label + '.png'
    out_meta = '0' * (6 - len(str(num))) + str(num) + '_' + label + '.json'
    out_path = os.path.join(render_path, out_path)
    print('output', out_path)
    render(out_path)
    meta = scene_params(mnist_filename)

    with open(os.path.join(meta_path, out_meta), 'wt') as f:
        json.dump(meta, f)



def make_obj(img):
    start = time.time()
    objs = mnist_util.make_mesh_voxels(img, threshold=0.12)
    end = time.time()
    print(img.channels)
    pixels = img.pixels


    bpy.ops.object.select_all(action='DESELECT')

    for obj in objs:
        obj.select_set(True)
    print('done in {0}'.format(end - start))

    bpy.ops.transform.rotate(value=math.radians(90), orient_axis='X')

    bpy.ops.transform.resize(value=(0.0656, 0.0656, 0.0656))

    bpy.ops.transform.rotate(value=math.radians(180), orient_axis='Z')

    target_loc = Vector((2.139021, 1.6, 0.7258))
    mean_coord = mnist_util.mean_coord(objs)

    translation = target_loc - mean_coord
    #translation = (1.588, 4.1, 1.349)
    bpy.ops.transform.translate(value=translation)


def render(out_path):
    bpy.context.scene.render.resolution_x = 1024 #perhaps set resolution in code
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.cycles.samples = 264
    bpy.data.scenes[0].render.engine = "CYCLES"

    #bpy.ops.object.delete()
    #collection = bpy.data.collections.get('mnist')
    #bpy.data.collections.remove(collection)
    bpy.context.scene.cycles.device = "GPU"
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

    # Set the device_type
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"

    bpy.context.scene.render.filepath = out_path

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
    bpy.ops.render.render(write_still=True)


def randomize_camera():
    bpy.ops.object.select_all(action='DESELECT')

    pivot = bpy.data.objects['pivot']
    pivot.select_set(True)

    bpy.context.view_layer.objects.active = pivot
    before = pivot.rotation_euler.copy()
    print(before)

    # move the pivot a bit
    shift = Vector((random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                    random.uniform(0.0, 0.1)))
    pivot.location += shift


    # rotate around the number
    rotate_angle = random.uniform(-0.5, 0.5) * math.pi * 2

    bpy.ops.transform.rotate(value=rotate_angle ,
        orient_axis='Z', orient_matrix_type='LOCAL',
        orient_type='LOCAL')


    # rotate up or down
    if random.random() < 0.5:
        rotate_angle_x = random.uniform(-0.1, 0) * math.pi * 2
        print('rotate up')
    else:
        print('rotate down')
        rotate_angle_x = random.uniform(0, 0.045) * math.pi * 2

    bpy.ops.transform.rotate(value=rotate_angle_x ,
        orient_axis='X', orient_matrix_type='LOCAL',
        orient_type='LOCAL')
    print('x rotation', rotate_angle_x / math.pi * 2)

    # move camera closer or farther away from the number
    scale = random.uniform(0, 1)
    if scale <= 0.5:
        scale += 0.4
    else:
        scale *= 2
    if 0 <= rotate_angle_x:
        scale = min(0.8, scale)
    print('scale', scale)
    bpy.ops.transform.resize(value=(scale, scale, scale))

    # adjust focal length
    bpy.data.cameras[0].lens *= scale

    return before


def to_list(mat):
    return list(list(x) for x in mat)


def scene_params(mnist_filename):
    rot = bpy.data.objects['Camera'].rotation_euler
    camera_rotation = rot.x, rot.y, rot.z
    pos = bpy.data.objects['Camera'].location
    pos = pos.x, pos.y, pos.z
    mean_obj = mnist_util.mean_coord1(bpy.data.collections.get('mnist').all_objects)
    result = dict()
    result['camera_euler'] = camera_rotation
    result['camera_pos'] = pos
    result['mean_number_pos'] = mean_obj.x, mean_obj.y, mean_obj.z
    result['source'] = mnist_filename
    result['K'] = to_list(mnist_util.get_calibration_matrix_K_from_blender(bpy.data.cameras[0]))
    result['RT'] = to_list(mnist_util.get_3x4_RT_matrix_from_blender(bpy.data.objects.get('Camera')))
    return result


if __name__ == '__main__':
    res = ["--meta=/home/leron/projects/mnist_saver/render_desc",
           "--render=/home/leron/projects/mnist_saver/render",
           "--source=/home/leron/projects/mnist_saver/train",
           "--blend=/home/leron/Documents/untitled.blend"]
    res = []
    for i, item in enumerate(sys.argv):
        if item == '--':
            res = sys.argv[i + 1:]; break


    print(res)
    parse = parse_args(res)
    for i in range(20000):
        # causes issues when run with gui
        bpy.ops.wm.open_mainfile(filepath=parse.blend)
        main(source_path=parse.source, meta_path=parse.meta, render_path=parse.render, num=len(os.listdir(parse.render)))
