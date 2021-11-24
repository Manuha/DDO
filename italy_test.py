from operator import matmul
import autograd
from autograd.builtins import tuple
import autograd.numpy as np
import pandas as pd

#Import ode solver and rename as BlackBox for consistency with blog
from scipy.integrate import odeint as BlackBox
import matplotlib.pyplot as plt
import copy


P = 59550000
rho = 0.9
num_days = 36
data = pd.read_csv('data/countries/italy.csv',header=0,index_col=0)
y_obs = data[["totale_positivi","dimessi_guariti","deceduti"]].iloc[:num_days]  # y_obs : seen in data
y_obs.columns=['I','R','D']
y_obs_np = y_obs.to_numpy()
I = y_obs_np[:,0]
R = y_obs_np[:,1]
D = y_obs_np[:,2]

N = len(y_obs)
t = range(N)
error_list = []

def get_susceptible(omega, alpha):
    I,R,D = y_obs_np.T
    pop = np.zeros(N,)
    pop.fill(P)
    S = np.round((((omega/alpha)*(pop))-I-R-D),0)
    
    return S

def f(y,t,alpha,beta,gamma,nu,omega):
    '''Function describing dynamics of the system'''
    I,R,D = y
    S = (((omega/alpha)*(P))-I-R-D)
    # ds = -beta*((S*I)/S+I)
    di = beta*((S*I)/S+I)-gamma*I-nu*I
    dr = gamma*I
    dd = alpha*nu*I

    return np.array([di,dr,dd])

#Jacobian wrt y
J = autograd.jacobian(f,argnum=0)
#Gradient wrt omega
grad_f_omega = autograd.jacobian(f,argnum=6)
grad_f_alpha = autograd.jacobian(f,argnum=2)

def get_weights(rho):
    powers = np.arange(1,N+1,1)
    base = np.zeros(N)
    base.fill(rho)
    weights = np.power(base,powers).reshape(N,1)[::-1]

    return weights


def compute_daily_increase(rho) -> np.array:
    y_dupe = copy.deepcopy(y_obs)
    y_shift = y_dupe.shift(1,fill_value=0)
    weights = get_weights(rho)
    y_diff = y_obs-y_shift
    return weights*np.array(y_diff)


def compute_phi(S,alpha,rho):
    Theta = N
    
    weights = get_weights(rho)
    
    phi = np.zeros((3*Theta, 3))
    i=0
    j = 0
    while i<Theta:
        phi[j,0] = weights[i]*(S[i]*y_obs_np[i,0])/ (S[i]+y_obs_np[i,0])
        phi[j,1] = -1*weights[i]*y_obs_np[i,0]
        phi[j,2] = -1*weights[i]*y_obs_np[i,0]/alpha
        
        phi[j+1,0] = 0
        phi[j+1,1] = weights[i]*y_obs_np[i,0]
        phi[j+1,2] = 0

        phi[j+2,0] = 0
        phi[j+2,1] = 0
        phi[j+2,2] = weights[i]*y_obs_np[i,0]
        i+=1
        j+=3
    
    return phi

delta = compute_daily_increase(0.9).reshape(-1,1)

def compute_rates(S,alpha,omega,rho):
    phi = compute_phi(S,alpha,rho)
    beta,gamma,nu = matmul(np.linalg.pinv(phi),delta)
    params = np.array([beta,gamma,nu])
    
    return phi,params

def ODESYS(Y,t,S,alpha,omega,rho):

    #Y will be length 4.
    #Y[0], Y[1] are the ODEs
    #Y[2], Y[3] are the sensitivities

    # Compute rates
    beta,gamma,nu = compute_rates(S,alpha,omega,rho)
    #ODE
    dy_dt = f(Y[0:3],t,alpha,beta,gamma,nu,omega).reshape(-1,)
    #Sensitivities
    jacobian_w_f = J(Y[:3],t,alpha,beta,gamma,nu,omega)
    grad_y_omega = (jacobian_w_f@Y[-3::] + grad_f_omega(Y[:3],t,alpha,beta,gamma,nu,omega)).reshape(-1,)
    grad_y_alpha = (jacobian_w_f@Y[-3::] + grad_f_alpha(Y[:3],t,alpha,beta,gamma,nu,omega)).reshape(-1,)
    
    return np.concatenate([dy_dt,grad_y_omega,grad_y_alpha])

def Cost(y_true):
    def cost(Y):
        
        '''Squared Error Loss'''
        n = y_true.shape[0]
        err = np.linalg.norm(y_true - Y, 'fro')**2
        # print("error")
        e = np.sum(err)/n
        print(e._value)
        error_list.append(e._value)
        return e
    return cost

def partial_phi_w_omega(Y,alpha,omega,rho):
    Theta = N
    S,I,R,D = Y
    weights = get_weights(rho)
    
    partial_phi = np.zeros((3*Theta, 3))
    i=0
    j = 0
    while i<Theta:
        partial_phi[j,0] = weights[i]*((alpha*P*(I[i]**2)) / (omega*P - alpha*(R[i]+D[i]))**2)
        partial_phi[j,1] = 0
        partial_phi[j,2] = 0
        
        partial_phi[j+1,0] = 0
        partial_phi[j+1,1] = 0
        partial_phi[j+1,2] = 0

        partial_phi[j+2,0] = 0
        partial_phi[j+2,1] = 0
        partial_phi[j+2,2] = 0
        i+=1
        j+=3
    
    return partial_phi


def partial_phi_w_alpha(Y,alpha,omega,rho):
    Theta = N
    S,I,R,D = Y
    weights = get_weights(rho)
    
    partial_phi = np.zeros((3*Theta, 3))
    i=0
    j = 0
    while i<Theta:
        partial_phi[j,0] = -weights[i]*((omega*P*(I[i]**2)) / (omega*P - alpha*(R[i]+D[i]))**2)
        partial_phi[j,1] = 0
        partial_phi[j,2] = 0
        
        partial_phi[j+1,0] = 0
        partial_phi[j+1,1] = 0
        partial_phi[j+1,2] = 0

        partial_phi[j+2,0] = 0
        partial_phi[j+2,1] = 0
        partial_phi[j+2,2] = I[i]/alpha
        i+=1
        j+=3
    
    return partial_phi


alpha = 10
omega = 0.2

def get_y_true(alpha,omega) -> np.array:
    y_true = copy.deepcopy(y_obs)
    y_true.columns=['I-','R-','D-']
    y_true['I-'] = alpha * y_true['I']

    return np.array(y_true)


prev_omega = omega_iter = 0.19
# prev_alpha = alpha_iter = 86.13193840334911
prev_alpha = alpha_iter = 87
iter = 0
alternating_error = []
maxiter = 200
while True:
    step_size_omega = 0.01 # Big steps
    step_size_alpha = 1 # Big steps
    dim_val = 0.9
    # i=0
    # while len(error_list)<2 or (len(error_list)>1 and (error_list[-2]-error_list[-1])>10):
    for i in range(maxiter):
        S = get_susceptible(omega_iter,alpha_iter )
        phi,params = compute_rates(S,alpha,omega,rho)
        diff_matrix = delta-np.matmul(phi,params)
        error = np.linalg.norm(diff_matrix,'fro')**2
        error_list.append(error)
        # step_size_alpha_diff = get_step_size_alpha(partial_w_alpha, partial_w_omega, )

        if iter%2==1:
            partial_w_omega = 2*np.matmul(diff_matrix.T, np.matmul(partial_phi_w_omega([S,I,R,D],alpha,omega,rho),params))[0,0]
            if len(error_list)>1 and error_list[-1]>=error_list[-2]:
                omega_iter = prev_omega 
                error_list.pop()
                step_size_omega *=dim_val
            else:
                # print(error_list)
                step_size_omega = 0.001
                prev_omega = omega_iter
            update_omega = step_size_omega*partial_w_omega
            omega_iter -=update_omega

            if omega_iter<=0 or omega_iter>=1:
                omega_iter = prev_omega 
                step_size_omega *=dim_val
        
        elif iter%2==0:
            partial_w_alpha = 2*np.matmul(diff_matrix.T, np.matmul(partial_phi_w_alpha([S,I,R,D],alpha,omega,rho),params))[0,0]
            if len(error_list)>1 and error_list[-1]>=error_list[-2]:
                alpha_iter = prev_alpha 
                error_list.pop()
                step_size_alpha *=dim_val
            else:
                # print(error_list)
                step_size_alpha = 1
                prev_alpha = alpha_iter
            update_alpha = step_size_alpha*partial_w_alpha
            alpha_iter -=update_alpha

            if alpha_iter<=0:
                alpha_iter = prev_alpha 
                step_size_alpha *=dim_val
        # i+=1

        if i%100==0:
            print(i,omega_iter,alpha_iter,error)
    # step_size_alpha *=dim_val
    # step_size_omega *=dim_val
    print(len(error_list)<2,(len(error_list)>1 and (error_list[-2]-error_list[-1])>10))
    print(iter,omega_iter,alpha_iter,error)
    print()
    plt.xlabel("Valid iteration")
    plt.ylabel("Error")
    if iter%2==0:
        plt.title('Minimization with optimal alpha')
    else:
        plt.title('Minimization with optimal omega')
    plt.plot(range(len(error_list)),error_list, label = 'error', linewidth = 5)
    plt.show()
    alternating_error.append(error_list[-1])
    # print(alternating_error[-1]," adf ",error_list[-1])
    if len(alternating_error)>1 and alternating_error[-2]-alternating_error[-1]<10:
        break
    iter+=1

print(error_list)
print(omega_iter,alpha_iter)
plt.plot(range(len(alternating_error)),alternating_error, label = 'alternating_error', linewidth = 5)
plt.show()

# sol = BlackBox(ODESYS, y0 = Y0, t = t, args = tuple([omega_iter,alpha_iter,rho]))
# true_sol = BlackBox(ODESYS, y0 = Y0, t = t, args = tuple([omega]))
# plt.plot(t,sol[:,1], label = 'I', color = 'C1', linewidth = 5)

# plt.scatter(t,y_true[:,0], marker = '.', alpha = 0.5)
# plt.scatter(t,y_true[:,1], marker = '.', alpha = 0.5)


# plt.plot(t,true_sol[:,0], label = 'Estimated ', color = 'k')
# plt.plot(t,true_sol[:,1], color = 'k')

# plt.show()

# for i in range(maxiter):
#     S = get_susceptible(omega_iter)
#     # Y0 = y_obs_np[0]
#     Y0 = np.concatenate([y_obs_np[0],[0,0,0],[0,0,0]])
#     # sol = BlackBox(ODESYS,y0 = Y0, t = t, args = tuple([S,alpha_iter,omega_iter,rho]))
#     sol = MyBlackBox(y0 = Y0, t = t, args = tuple([S,alpha_iter,omega_iter,rho]))
#     grad_C = autograd.grad(cost)

#     Y = sol[:,:3]

#     cost_gradient = grad_C(Y)
#     if len(error_list)>1 and error_list[-1]>error_list[-2]:
#         prev_omega = omega_iter
#         prev_alpha = alpha_iter
#         error_list.pop()

#     update_alpha = step_size_alpha*(cost_gradient*sol[:,-3:]).sum()
#     update_omega = step_size_omega*(cost_gradient*sol[:,-6:-3]).sum()

#     alpha_iter -=update_alpha
#     omega_iter -=update_omega
#     # if update_alpha>0:
#     #     alpha_iter -=min(1,abs(update_alpha))
#     # else:
#     #     alpha_iter +=min(1,abs(update_alpha))
#     # if update_omega>0:
#     #     omega_iter -=min(0.001,abs(update_omega))
#     # else:
#     #     omega_iter +=min(0.001,abs(update_omega))

#     if alpha_iter<=0 or omega_iter<=0:
#         prev_omega = omega_iter
#         prev_alpha = alpha_iter
#         error_list.pop()
#     if i%10==0:
#         print(i,omega_iter,alpha_iter)
#     step_size_alpha *=dim_val
#     step_size_omega *=dim_val
