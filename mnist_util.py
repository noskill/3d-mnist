import bpy
import bmesh
import itertools
from mathutils import Matrix, Vector


def get_pix(img, x, y):
    target = [x, y] # X, Y
    w, h = img.size
    if x < 0 or y < 0 or w <= x or h <= y:
        return [0, 0, 0, 0]
    # target.Y * image width + target.X * 4 (as the table Pixels contains separate RGBA values)
    index = ( target[1] * w + target[0] ) * img.channels

    # aggregate the read pixel values into a nice array
    pixel = [
        img.pixels[index],     # RED
        img.pixels[index + 1], # GREEN
        img.pixels[index + 2], # BLUE
        img.pixels[index + 3]  # ALPHA
    ]

    return pixel


def compute_point(img, x, y, scale=2):
    # compute height by averaging brightness
    acc = 0.0
    for (d1, d2) in itertools.product((1,-1), (1, -1)):
        acc += max(get_pix(img, x + d1, y + d2)[:3]) / 4
    return x, y, (acc + 0.5) ** scale


def populate(bm, img, threshold):
    # by pixel coords
    faces = dict()
    # index by face coords
    vert_set = set()
    vertices = dict() # coord -> bmesh vert
    w, h = img.size
    for x in range(w):
        for y in range(h):
            pix = get_pix(img, x, y)
            brightness = max(pix[:3])
            if brightness > threshold:
                # 4 points
                p1 = compute_point(img, x, y)
                p2 = compute_point(img, x, y + 1)
                p3 = compute_point(img, x + 1, y)
                p4 = compute_point(img, x + 1, y + 1)
                for_face = [] # List[bmesh.types.BMVert]
                for p in [p1, p2, p3, p4]:
                    if p not in vertices:
                        vert = list(bmesh.ops.create_vert(bm, co=p).values())[0][0]
                        vertices[p] = vert
                        for_face.append(vert)
                    else:
                        for_face.append(vertices[p])
                face = bmesh.ops.contextual_create(bm, geom=for_face)


class MnistDrawer:

    def __init__(self):
        self.bm = None

    def make_vert(self, x, y, z):
        p = x, y, z
        vert = list(bmesh.ops.create_vert(self.bm, co=p).values())[0][0]
        return vert

    def create_cube(self, height):
        bottom = []
        for (d1, d2) in [0, 0], [1, 0], [1, 1], [0, 1]:
            v = self.make_vert(d1, d2, 0)
            bottom.append(v)

        face = bmesh.ops.contextual_create(self.bm, geom=bottom)['faces']

        extruded = bmesh.ops.extrude_discrete_faces(self.bm, faces=face)['faces'][0]

        for v in extruded.verts:
            v.co.z = height

    def populate_vox(self, img, threshold):
        mycol = bpy.data.collections.new('mnist')
        bpy.context.scene.collection.children.link(mycol)
        w, h = img.size

        template_ob = self.process(0, 0, 1, mycol)
        bpy.ops.object.select_all(action='DESELECT')
        objs = []
        for x in range(w):
            for y in range(h):
                pix = get_pix(img, x, y)
                brightness = max(pix[:3])
                if brightness > threshold:
                    obj = template_ob.copy()
                    bpy.data.collections[mycol.name].objects.link(obj)
                    obj.location = (x - w // 2, y - h // 2, 0)
                    obj.select_set(True)
                    bpy.ops.transform.resize(value=(1, 1, brightness))
                    obj.select_set(False)
                    objs.append(obj)
        bpy.data.collections[mycol.name].objects.unlink(template_ob)
        return objs

    def process(self, x, y, brightness, mycol=None):
        self.bm = bmesh.new()   # create an empty BMesh
        self.create_cube(brightness)
        new_mesh = bpy.data.meshes.new('new_mesh')
        self.bm.to_mesh(new_mesh)
        self.bm.free()  # free and prevent further access
        self.bm = None
        new_object = bpy.data.objects.new('new_object', new_mesh)
        obj1 = mirror(new_object)
        join(new_object, obj1)
        bpy.data.collections[mycol.name].objects.link(new_object)
        normals(new_object)
        material(new_object)
        return new_object


def material(ob):
    mat = bpy.data.materials.get("MnistMaterial")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="MnistMaterial")
    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)


def normals(new_object):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = new_object
    new_object.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    # select al faces
    bpy.ops.mesh.select_all(action='SELECT')
    # recalculate outside normals
    bpy.ops.mesh.normals_make_consistent(inside=False)
    # go object mode again
    bpy.ops.object.editmode_toggle()


def make_mesh_voxels(img, threshold=0.05):
    drawer = MnistDrawer()
    objs = drawer.populate_vox(img, threshold=threshold)
    return objs


def cleanup(obj):
    merge_threshold = 0.001
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold = merge_threshold)


def join(*obj):
    ctx = bpy.context.copy()
    # one of the objects to join
    ctx['active_object'] = obj[0]
    ctx['selected_editable_objects'] = obj
    bpy.ops.object.join(ctx)

def make_mesh_flat(img, threshold=0.05):


    # Get a BMesh representation
    bm = bmesh.new()   # create an empty BMesh

    # Finish up, write the bmesh back to the mesh
    populate(bm, img, threshold=threshold)
    for v in bm.verts:
        v.co.x -= 14
        v.co.y -= 14


    all_boundary = []
    for edge in bm.edges:
        if edge.is_boundary:
            all_boundary.append(edge)

    geom = bmesh.ops.extrude_edge_only(bm, edges=all_boundary)['geom']
    verts = [x for x in geom if isinstance(x, bmesh.types.BMVert)]
    for vert in verts:
        vert.co.z = 0


    face = [x for x in bm.faces]

    # there are some issues with this approach
    #bmesh.ops.mirror(bm, geom=face, axis='Z', merge_dist=0)

    new_mesh = bpy.data.meshes.new('new_mesh')
    bm.to_mesh(new_mesh)
    bm.free()  # free and prevent further access
    new_object = bpy.data.objects.new('new_object', new_mesh)
    bpy.context.scene.collection.children[0].objects.link(new_object)

    new_object.select_set(True)
    mirror(new_object)


def mirror(new_object):
    new_obj = new_object.copy()
    new_obj.data = new_object.data.copy()
    bpy.context.scene.collection.children[0].objects.link(new_obj)

    new_object.select_set(False)

    C = bpy.context

    new_obj.select_set(True)
    C.view_layer.objects.active = new_obj
    assert(new_obj is not None)

    bpy.ops.transform.resize(value=(1, 1, -1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, )
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return new_obj


def mean_coord(objs):
    coords = [0, 0, 1000]
    for obj in objs:
        x, y, z = obj.location
        coords[0] += x
        coords[1] += y
        # need to align to floor level
        coords[2] = min(z, coords[2])

    coords[0] /= len(objs)
    coords[1] /= len(objs)
    return Vector(coords)


def mean_coord1(objs):
    coords = [0, 0, 1000]
    for obj in objs:
        x, y, z = obj.location
        coords[0] += x
        coords[1] += y
        coords[2] += z

    coords[0] /= len(objs)
    coords[1] /= len(objs)
    coords[2] /= len(objs)
    return Vector(coords)


# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K


# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT