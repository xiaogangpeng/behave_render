import numpy as np
import bpy

from .materials import body_material

from .materials import obj_material

from .materials import floor_mat

# green
# GT_SMPL = body_material(0.009, 0.214, 0.029)
# GT_SMPL = body_material(0.035, 0.415, 0.122)
GT_SMPL = body_material(0.274, 0.674, 0.792)


# blue
# GEN_SMPL = body_material(0.022, 0.129, 0.439)
# Blues => cmap(0.87)
# GEN_SMPL = body_material(0.035, 0.322, 0.615)
# Oranges => cmap(0.87)

# (0.658, 0.214, 0.0114) ORANGE 

GEN_SMPL = body_material(0.274, 0.674, 0.792)

BLACK_MAT = body_material(0.0, 0.0, 0.0)
WHITE_MAT = body_material(1.0, 1.0, 1.0)

class Meshes:
    def __init__(self, h_data, o_data, h_data_path, o_data_path, *, gt, mode, canonicalize, always_on_floor, oldrender=False, **kwargs):
        
        # data = np.load(h_data_path, allow_pickle=True).item()['vertices']
        # obj_data = np.load(o_data_path, allow_pickle=True).item()['vertices']
        # data = data.permute(2, 0, 1).numpy()
        # obj_data = obj_data.view(np.ndarray).transpose(2, 0, 1)
        
        self.faces = np.load(h_data_path, allow_pickle=True).item()['faces']
        self.obj_faces = np.load(o_data_path, allow_pickle=True).item()['faces']

        self.h_contact_idx = np.load(h_data_path, allow_pickle=True).item()['contact_idx'].transpose(1,0)
        self.o_contact_idx = np.load(o_data_path, allow_pickle=True).item()['contact_idx'].transpose(1,0)  #  N, T -> T, N
        
        data = prepare_meshes(h_data, canonicalize=canonicalize,
                              always_on_floor=True)
        
        obj_data = prepare_obj_meshes(o_data, canonicalize=canonicalize,
                              always_on_floor=True)

        
        self.data = data
        self.obj_data = obj_data
        self.mode = mode
        self.oldrender = oldrender

        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)

        if gt:
            self.mat = GT_SMPL
            self.mat2 = GEN_SMPL
        else:
            self.mat = GEN_SMPL
            self.mat2 = GT_SMPL

    def get_sequence_mat(self, frac):
        import matplotlib
        # cmap = matplotlib.cm.get_cmap('Blues')
        cmap = matplotlib.cm.get_cmap('Oranges')
        cmap2 = matplotlib.cm.get_cmap('Blues')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end-begin)*frac)
        rgbcolor2 = cmap2(begin + (end-begin)*frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        mat2 = obj_material(*rgbcolor2, oldrender=self.oldrender)
        
        return mat, mat2
    
    
    

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        
        h_contact_names = []
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)
        
        idx =  np.where(self.h_contact_idx[index]==1.0)[0]
        choose_v = vertices[idx,:]
        # #create spheres
        if len(choose_v>0):
            for i in range(len(choose_v)):
                h_contact_name = f"h_{str(index).zfill(4)}"
                bpy.ops.mesh.primitive_uv_sphere_add(segments = 64, radius=0.01, ring_count = 32, location=(float(choose_v[i][0]),float(choose_v[i][1]),float(choose_v[i][2])))
                bpy.ops.object.shade_smooth()
                obj = bpy.data.objects["Sphere"]                                     
                obj.name = h_contact_name
                obj.data.name = h_contact_name
                obj.active_material = BLACK_MAT
                # bpy.ops.object.select_all(action='DESELECT')
                # obj.select_set(True)
                # obj.active_material = BLACK_MAT
            h_contact_names.extend(h_contact_name)

        
        # contact = self.h_contact_idx[index]
        # print(f"==============verticve :{idx}")

        return name, h_contact_names
    
    def load_obj_in_blender(self, index, mat):
        vertices = self.obj_data[index]
        name = f"obj_{str(index).zfill(4)}"
        faces = self.obj_faces
        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)
        
        idx =  np.where(self.o_contact_idx[index]==1.0)[0]
        choose_v = vertices[idx,:]
        o_contact_names = []
        # #create spheres
        if len(choose_v>0):
            for i in range(len(choose_v)):
                o_contact_name = f"h_{str(index).zfill(4)}"
                bpy.ops.mesh.primitive_uv_sphere_add(segments = 64, radius=0.01, ring_count = 32, location=(float(choose_v[i][0]),float(choose_v[i][1]),float(choose_v[i][2])))
                bpy.ops.object.shade_smooth()
                obj = bpy.data.objects["Sphere"]                                     
                obj.name = o_contact_name
                obj.data.name = o_contact_name
                           
                obj.active_material = WHITE_MAT
                # bpy.ops.object.select_all(action='DESELECT')
                # obj.select_set(True)
                # obj.active_material = BLACK_MAT
            o_contact_names.extend(o_contact_name)

        
        # contact = self.h_contact_idx[index]
        # print(f"==============verticve :{idx}")

        return name, o_contact_names
        
        return name

    def __len__(self):
        return self.N


def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fitted mesh do not need fixing axis
    # # fix axis
    # data[..., 1] = - data[..., 1]
    # data[..., 0] = - data[..., 0]

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data

def prepare_obj_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fitted mesh do not need fixing axis
    # # fix axis
    # data[..., 1] = - data[..., 1]
    # data[..., 0] = - data[..., 0]

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # # Remove the floor
    # data[..., 2] -= data[..., 2].min()

    # # Put all the body on the floor
    # if always_on_floor:
    #     data[..., 2] -= data[..., 2].min(1)[:, None]

    return data
