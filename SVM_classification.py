import numpy as np
import scipy.optimize

data = np.load("Wine_Data.npz")
X_training = data['X_training']
y_training = data['y_training']

X_test = data['X_test']
y_test = data ['y_test']

'''
Solved following quadratic problem:
min    1/2 x^T*H*x + f^T*x
s.t.   A*x <= b
Compare with: https://www.mathworks.com/help/optim/ug/quadprog.html
'''
def quadprog(H, f, A, b):
    def g(x):
        return (0.5 * np.dot(x, H).dot(x) + np.dot(f, x))

    constraints = [{
		'type': 'ineq',
		'fun': lambda x, A=A, b=b, i=i: -((np.dot(A, x) - b)[i])
	} for i in range(A.shape[0])]

    result = scipy.optimize.minimize(g, x0=np.zeros(len(H)), method='SLSQP',
        constraints=constraints, tol=1e-10)
    return result

#Set Parameters
lam = 0.5
m,n = X_training.shape

#Translate problem into quadprog format
H = 2*np.block([[lam*np.eye(n), np.zeros((n, m+1))], [np.zeros((m+1, n+1+m))]])
f = np.concatenate((np.zeros(n+1),(1.0/m)*np.ones(m)))
A = np.block([[-np.diag(y_training)@X_training, np.array([-y_training]).T, -np.eye(m)],[np.zeros((m,n+1)),-np.eye(m)]])
b = np.concatenate((-np.ones(m),np.zeros(m)))

#Solve problem
sol = quadprog(H,f,A,b).x;

#Read parameters from solution
w = sol[0:n]
b = sol[n]
e = sol[n+1:n+m+1]

#Test prediction on training set
m_test, n_test = X_test.shape
y_prediction = np.sign(X_test@w + b)
z = np.diag(y_test) @ y_prediction
success = 100*len(z[z>0])/m_test

print(success, "% der Testpunkte wurden korrekt klassifiziert.")
