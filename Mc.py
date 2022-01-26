import numpy as np
import matplotlib.pyplot as plt 
import statistics as sts
#from scipy import linalg
N=15
x=np.linspace(0,5,N)
y=(x-2)**2+np.random.randn(15)

k=5
A = []
for i in range(N):
    b=[]
    for j in range(k+1):
        b.insert(0,x[i]**j)
    A.append(b)
A=np.array(A)

what=np.array(np.linalg.lstsq(A, y, rcond = None)[0])


# evaluate the polynomial on evenly spaced points.
# NOTE: you can also use the command yy = polyval(what,xx);

xx = np.linspace(0,5,200)
AA = [];
for i in range(k+1):
    AA.append(xx**(k-i-1))

AA=(np.array(AA)).transpose()

yy = np.dot(AA,what)

# plot the data points and the best fit
plt.scatter(x,y)
plt.plot(xx,yy, 'b-')
plt.axis([0, 6, 0, 20])
plt.title('best fit using a polynomial of degree',k)
plt.show()


#err = np.linalg.norm(y - np.dot(A,what))

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

k=5
kmax=5
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
Z = [errcv[i] for i in range(5)]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Y1, 0.4, label = 'average test error')
plt.bar(X_axis + 0.2, Z, 0.4, label = 'err')
  
plt.xticks(X_axis, X)
plt.xlabel("polynomial degree")
plt.ylabel("average test error")
plt.legend()
plt.show()

'''A = np.empty((15,6))
for i in range(k,-1,-1):
    X=np.empty((15,1))
    X=x**i
    A.concatenate(X)
'''
'''

import 


figure(2); clf
bar([err avg_err_cv])
xlabel('polynomial degree')
ylabel('average test error')
legend('performance on training data','using cross-validation',...
    'Location','north')
'''
