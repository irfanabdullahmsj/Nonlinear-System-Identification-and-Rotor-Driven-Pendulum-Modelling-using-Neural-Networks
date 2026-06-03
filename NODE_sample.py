# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Import the ODE solver from torchdiffeq
from torchdiffeq import odeint
from scipy.io import loadmat

# Load the data from the .mat file
data = loadmat(r"I:\h_da\Summer 2026\Team_Project\Data_Preprocessing\Reduced_Exp_12May_1457_upsamp_filt.mat")

# Extract the relevant variables from the loaded data
phi = data['PA']
phid = data['PAD']
u = data['u']
t = data['t']

# Squeeze the arrays to remove any singleton dimensions
phi = phi.squeeze()
phid = phid.squeeze()
u = u.squeeze()
t = t.squeeze()

# Plot the angular position
plt.figure()
plt.plot(t, phi, label='phi')
plt.title('Angular_Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('phi')
plt.grid()

# Plot the angular velocity
plt.figure()
plt.plot(t, phid, label='phid')
plt.title('Angular_Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('phid')
plt.grid()


# Plot the control input
plt.figure()
plt.plot(t, u, label='u')
plt.title('Control_Input vs Time')
plt.xlabel('Time (s)')
plt.ylabel('phi')
plt.grid()
#plt.show()

# Calculate the mean and standard deviation for phi, phid, and u
phi_mean = np.mean(phi)
phi_std = np.std(phi)
phid_mean = np.mean(phid)
phid_std = np.std(phid)
u_mean = np.mean(u)
u_std = np.std(u)

#Sample the data to reduce the number of points for training
#change skip value to adjust the number of data points used for training (e.g., skip=10 for every 10th point)
skip =1
phi_ = phi[::skip]
phid_ = phid[::skip]
u_ = u[::skip]
t_ = t[::skip]

#uncomment the following lines to print the number of data points after sampling
#print("Number of phi data points  :", len(phi))
#print("Number of phid data points :", len(phid))
#print("Number of u data points    :", len(u))
#print("Number of t data points    :", len(t))


# Normalize the data using the calculated mean and standard deviation
phi_normalized = (phi_ - phi_mean) / phi_std
phid_normalized = (phid_ - phid_mean) / phid_std
u_normalized = (u_ - u_mean) / u_std

# Stack the normalized phi and phid to create the input data for the NODE model
x=np.stack(
    (phi_normalized, phid_normalized), axis=1
    )

# Convert the input data and control input to PyTorch tensors
x_tensor = torch.tensor(
    x, dtype=torch.float32
)
u_tensor = torch.tensor(
    u_normalized, dtype=torch.float32
).unsqueeze(1)  # Add an extra dimension for the control input
t_tensor = torch.tensor(
    t_, dtype=torch.float32
)

# Split the data into training and testing sets (80% training, 20% testing)
N=len(t_tensor)
split_idx = int(0.8 * N)
x_train = x_tensor[:split_idx]
u_train = u_tensor[:split_idx]
t_train = t_tensor[:split_idx]
x_test = x_tensor[split_idx:]
u_test = u_tensor[split_idx:]
t_test = t_tensor[split_idx:]

#Batch the training data into windows for better training stability
window_size = 200
stride = 50

x_windows = []
u_windows = []
t_windows = []

for i in range(0, len(x_train) - window_size, stride):

    x_w = x_train[i:i+window_size]
    u_w = u_train[i:i+window_size]
    t_w = t_train[i:i+window_size]

    # make local time start from zero
    t_w = t_w - t_w[0]

    x_windows.append(x_w)
    u_windows.append(u_w)
    t_windows.append(t_w)


# Initial state (phi and phid at time t=0)
x0_train = x_train[0].unsqueeze(0)  

#Sample time step
dt = t_[1] - t_[0]  # Time step

# Define the ODE function for the NODE model, which takes into account the control input
# Adjust the hidden layer sizes and activation functions as needed for better performance
class ODEFunc(nn.Module):
    def __init__(self,u_data):
        super().__init__()
        self.u_data = u_data 
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # Input: [phi, phid, u]
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)    # Output: [dphi/dt, dphid/dt]
        )
    def forward(self, t, x):
        idx = min(
            int(round(t.item() / dt)),
            len(self.u_data) - 1
        )
        u_current = self.u_data[idx].unsqueeze(0)  # Get the control input at time t
        x_u = torch.cat(
            [x, u_current], 
            dim=1
        )  # Concatenate state and control input
        dxdt = self.net(x_u)
        return dxdt

# Create NODE function instance
func = ODEFunc(u_data=u_train)  # Pass the training control input to the ODE function

# Define the optimizer for training the NODE model
optimizer = torch.optim.Adam(
    func.parameters(), 
    lr=1e-3
    )

# Test the initial prediction before training
pred_test = odeint(
    func, 
    x0_train, 
    t_train,
    method='rk4' # Use the Runge-Kutta 4th order method for better accuracy
)

# Training loop for the NODE model
# Increase the number of epochs for better convergence, but be mindful of overfitting
epochs = 300
loss_history = []
for epoch in range(epochs):
    optimizer.zero_grad()
    pred_x = odeint(
        func, 
        x0_train, 
        t_train,
        method='rk4' # Use the Runge-Kutta 4th order method for better accuracy
    )
    pred_x = pred_x.squeeze(1)  # Remove the extra dimension
    loss = torch.mean((pred_x - x_train) ** 2)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(epoch, loss.item())

# Plot the training loss over epochs
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid()
#plt.show()

# Update the ODE function with the test control input for evaluation
func.u_data = u_test  # Update the ODE function with the test control input
x0_test = x_test[0].unsqueeze(0)  # Initial state for testing
with torch.no_grad():
    pred_test = odeint(
        func, 
        x0_test, 
        t_test,
        method='rk4' # Use the Runge-Kutta 4th order method for better accuracy
    )
pred_test = pred_test.squeeze(1).numpy()  # Remove the extra dimension
phi_pred = pred_test[:, 0] 
phid_pred = pred_test[:, 1]

#denormalize the predictions
phi_pred_denorm = phi_pred * phi_std + phi_mean
phid_pred_denorm = phid_pred * phid_std + phid_mean

# Plot the true vs predicted angular position
plt.figure(figsize=(10,5))
plt.plot(
    t_[split_idx:],
    phi_[split_idx:],
    label='True'
)
plt.plot(
    t_[split_idx:],
    phi_pred_denorm,
    '--',
    label='NODE Prediction'
)
plt.xlabel('Time')
plt.ylabel('Angle')
plt.title('Testing Prediction')
plt.legend()
plt.grid()
#plt.show()

# Plot the true vs predicted angular velocity
plt.figure(figsize=(10,5))
plt.plot(
    t_[split_idx:],
    phid_[split_idx:],
    label='True Velocity'
)
plt.plot(
    t_[split_idx:],
    phid_pred_denorm,
    '--',
    label='NODE Velocity'
)
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.title('Velocity Prediction')
plt.legend()
plt.grid()
plt.show()