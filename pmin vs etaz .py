from pylab import*
eta = 0


ee = []

pp = []

for i in range(50 , 5000):
    
    eta = i/100
    a = eta/(eta+1)
    p = 1/(1+a)
    ee.append(eta)
    pp.append(p)
    
plot(ee, pp)
xlim(0.500 , 50 )
ylim(0, 1)
xlabel('eta z')
ylabel('Pmin')