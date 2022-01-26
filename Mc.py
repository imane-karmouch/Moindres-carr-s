import numpy as np
import matplotlib.pyplot as plt 
import statistics as sts

N=15
x =np.array((np.linspace(0,5,N))).T
y=(x-2)**2+np.random.randn(15)

k=5
A = []
for i in range(k,-1,-1):
  A.append((x**(i)))
A = (np.array(A)).transpose

what=(np.linalg.lstsq(A, y, rcond = None)[0])
# evaluate the polynomial on evenly spaced points.
xx = np.array(np.linspace(0,5,200)).T
AA = []
for i in range(k+1):
    AA.append(xx**(k-i))
AA=(np.array(AA)).transpose()
yy = np.dot(AA,what)

# plot the data points and the best fit

plt.scatter(x, y)
plt.plot(xx , yy , 'b')
plt.title("best fit using a polynomial of degree "+str(k)) 
plt.show()
err1 = np.linalg.norm(y - np.dot(A,what))

#NO CROSS-VALIDATAION
kmax = 5

# generate one large A matrix. We can get the matrices for particular k
# values by just picking the correct columns from this matrix.
A = []
for i in range(N):
    b=[]
    for j in range(kmax+1):
        b.insert(0,x[i]**j)
    A.append(b)
A=np.array(A)

err = np.zeros(kmax,);

for k in range(1,kmax+1):
    Amat = np.array(A[:,kmax-k:kmax+1])
    what = np.array(np.linalg.lstsq(Amat, y, rcond = None)[0]) 
    # compute the error and divide by the number of points
    err[k-1]= (np.linalg.norm(y - np.dot(Amat,what)))/N #for each k
    
#USIGN CROSS-VALIDATION

T = 12            # number of training points to use
trials = 1000      # number of averaging trials

# generate one large A matrix. We can get the matrices for particular k
# values by just picking the correct columns from this matrix.
A = [];
for i in range(k+1):
  A.append((x**(k-i)))
A = (np.array(A)).transpose()

errcv = np.zeros((kmax,trials))

for t in range( 1,trials+1):
    r=np.random.permutation(N)
    train = [r[i] for i in range(0,T)]
    test = [r[i] for i in range(T,len(r))]

    for k in range(1,kmax+1):
        
        Atrain = A[train,kmax-k:kmax+1]
        Atest = A[test,kmax-k:kmax+1]

        ytrain = y[train]
        ytest = y[test]

        what = np.array(np.linalg.lstsq(Atrain, ytrain, rcond = None)[0]) 
        # compute error and divide by the number of test points
        errcv[k-1,t-1] = np.linalg.norm(ytest - np.dot(Atest,what)) / (N-T)


avg_err_cv=[]
for i in range(5):
    avg_err_cv.append(sts.mean(errcv[i,:]))

# compare the performance of the least-squares on training data to the
# performance when using cross-validation

X = [1,2,3,4,5]
Y1 = [avg_err_cv[i] for i in range(5)]
y2= [err[i] for i in range(5)]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Y1, 0.4, label = 'using cross_validation')
plt.bar(X_axis + 0.2, y2, 0.4, label = 'performance on training data')
  
plt.xticks(X_axis, X)
plt.xlabel("polynomial degree")
plt.ylabel("average test error")
plt.legend()
plt.show()
