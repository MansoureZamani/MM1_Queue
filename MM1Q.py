import matplotlib.pyplot as plt
import numpy
import statistics

MU = 1.0
LAMBDAS = numpy.arange(0.01, 0.96, 0.02).tolist()
# little formula : N = λ * T
N = numpy.zeros(len(LAMBDAS)).tolist()
T = numpy.zeros(len(LAMBDAS)).tolist()
NPKT = 10000  # Number of Packet which are served
i = 0

for lam in LAMBDAS:
    now = 0
    systemState = [0] * (NPKT + 1)
    departureTime = [0] * (NPKT + 1)
    npktServed = 0
    interArrivalTime = numpy.random.exponential(scale=1 / lam, size=(NPKT + 1))
    arrivalTime = numpy.cumsum(interArrivalTime)
    serviceTime = numpy.random.exponential(scale=1 / MU, size=(NPKT + 2))
    nextDeparture = float('inf')
    nextArrival = arrivalTime[1]
    npktArrived = 1
    currentLength = 0

    while npktServed < NPKT:
        if nextArrival < nextDeparture:
            now = nextArrival
            systemState[npktArrived] = currentLength
            currentLength += 1
            if currentLength == 1:
                nextDeparture = now + serviceTime[npktServed + 1]
            if npktArrived < NPKT:
                npktArrived += 1
                nextArrival = arrivalTime[npktArrived]
            else:
                nextArrival = float('inf')

        else:
            now = nextDeparture
            currentLength -= 1
            npktServed += 1
            departureTime[npktServed] = now
            if currentLength > 0:
                nextDeparture = now + serviceTime[npktServed + 1]
            else:
                nextDeparture = float('inf')

    N[i] = statistics.mean(systemState)
    temp = [0] * (NPKT + 2)
    for j in range(NPKT):
        temp[j] = departureTime[j+1] - arrivalTime[j+1] - serviceTime[j+1]
    T[i] = numpy.sum(temp) / 1000
    i += 1

Tananlysis = 1 / (MU - numpy.asarray(LAMBDAS))
plt.plot(LAMBDAS, Tananlysis)
plt.plot(LAMBDAS, T)
plt.xlabel('ρ')
plt.ylabel('delay')
plt.title('Delay - Rho')
plt.show()

Nananlysis = numpy.asarray(LAMBDAS) / (MU - numpy.asarray(LAMBDAS))
plt.plot(LAMBDAS, Nananlysis)
plt.plot(LAMBDAS, N)
plt.xlabel('ρ')
plt.ylabel('N')
plt.title('N - Rho')
plt.show()
