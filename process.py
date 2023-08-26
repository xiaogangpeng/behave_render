import numpy as np
# data = np.load("/home/ericpeng/DeepLearning/Projects/human_motion_generation/motion-latent-diffusion/sample04_rep00_obj_params.npy", allow_pickle=True).item()
# data = data['vertices'].view(np.ndarray).transpose(2,0, 1)
# data = np.load("/home/ericpeng/DeepLearning/Projects/human_motion_generation/motion-latent-diffusion/template/010632_mesh.npy")
# print(f"data {data.shape}")
# print(f"te:{data['motion'].shape}    {data['vertices'].shape}")
# np.save("/home/ericpeng/DeepLearning/Projects/human_motion_generation/motion-latent-diffusion/template/sample_04_obj_mesh.npy", data)


data = np.load("/home/ericpeng/DeepLearning/Projects/human_motion_generation/motion-latent-diffusion/template/sample00_rep00_smpl_params.npy", allow_pickle=True).item()['text']
print(f"data :{data}")










