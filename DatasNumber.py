# %%
# Import library

from scipy.io import loadmat


# %%
# Load MAT file

data = loadmat(
    r"I:\h_da\Summer 2026\Team_Project\Datasets\NODE\Exp_12May_1503_upsamp_filt.mat"
)

print(data.keys())


# %%
# Extract variables

phi = data['PA'].squeeze()
phid = data['PAD'].squeeze()
u = data['u'].squeeze()
t = data['t'].squeeze()


# %%
# Print number of data points

print("Number of phi data points  :", len(phi))
print("Number of phid data points :", len(phid))
print("Number of u data points    :", len(u))
print("Number of t data points    :", len(t))

skip =1
phi_ = phi[::skip]
phid_ = phid[::skip]
u_ = u[::skip]
t_ = t[::skip]

print("Number of phi_ data points  :", len(phi_))
print("Number of phid_ data points :", len(phid_))  
print("Number of u_ data points    :", len(u_))
print("Number of t_ data points    :", len(t_))