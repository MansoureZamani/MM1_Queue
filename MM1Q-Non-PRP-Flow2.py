import matplotlib.pyplot as plt
import numpy
import statistics

MU = 1.0
LAMBDAS = numpy.arange(0.01, 0.96, 0.02).tolist()
ConstLambda = 0.01
# little formula : N = λ * T
N1 = numpy.zeros(len(LAMBDAS)).tolist()
T1 = numpy.zeros(len(LAMBDAS)).tolist()
N2 = numpy.zeros(len(LAMBDAS)).tolist()
T2 = numpy.zeros(len(LAMBDAS)).tolist()

NPKT = 500  # Number of Packet which are served
i = 0

for lam in LAMBDAS:
    now = 0
    systemState_Flow1 = [0] * (NPKT + 1)
    systemState_Flow2 = [0] * (NPKT + 1)
    departureTime_Flow1 = [0] * (NPKT + 1)
    departureTime_Flow2 = [0] * (NPKT + 1)
    npktServed = 0
    ArrivalTime_Flow1 = numpy.random.exponential(scale=1 / ConstLambda, size=(NPKT + 1))
    numpy.random.shuffle(ArrivalTime_Flow1)

    ArrivalTime_Flow1 = numpy.cumsum(ArrivalTime_Flow1)

    ArrivalTime_Flow2 = numpy.random.exponential(scale=1 / lam, size=(NPKT + 1))
    numpy.random.shuffle(ArrivalTime_Flow2)
    ArrivalTime_Flow2 = numpy.cumsum(ArrivalTime_Flow2)

    serviceTime_Flow1 = numpy.random.exponential(scale=1 / MU, size=(NPKT + 1))
    serviceTime_Flow2 = numpy.random.exponential(scale=1 / MU, size=(NPKT + 1))

    nextDeparture = float('inf')
    nextArrival_Flow2 = ArrivalTime_Flow2[1]
    nextArrival_Flow1 = ArrivalTime_Flow1[1]
    if nextArrival_Flow1 < nextArrival_Flow2:
        nextDeparture = nextArrival_Flow1
    else:
        nextDeparture = nextArrival_Flow2

    npktArrived_Flow1 = 1
    npktArrived_Flow2 = 1
    npktServed_Flow1 = 0
    npktServed_Flow2 = 0
    serverIsBusy_Flow1 = False
    serverIsBusy_Flow2 = False
    queue_Flow1 = []
    queue_Flow2 = []
    currentArrival_Flow2 = 0
    currentDeparture_Flow1 = 0
    while npktServed < 2 * NPKT:
        if nextArrival_Flow1 != float('inf') or nextArrival_Flow2 != float('inf'):
            if nextArrival_Flow1 < nextArrival_Flow2:
                queue_Flow1.append((ArrivalTime_Flow1[npktArrived_Flow1], serviceTime_Flow1[npktArrived_Flow1]))
                val = numpy.searchsorted(ArrivalTime_Flow1, departureTime_Flow1[npktServed_Flow1])
                systemState_Flow1[npktArrived_Flow1] = val - npktServed_Flow1
                if npktArrived_Flow1 < NPKT:
                    npktArrived_Flow1 += 1
                    nextArrival_Flow1 = ArrivalTime_Flow1[npktArrived_Flow1]
                else:
                    nextArrival_Flow1 = float('inf')
            else:
                queue_Flow2.append((ArrivalTime_Flow2[npktArrived_Flow2], serviceTime_Flow2[npktArrived_Flow2]))
                val = numpy.searchsorted(ArrivalTime_Flow1, departureTime_Flow2[npktServed_Flow2])
                val2 = numpy.searchsorted(ArrivalTime_Flow2, departureTime_Flow2[npktServed_Flow2])
                systemState_Flow2[npktArrived_Flow2] = val + val2 - npktServed_Flow2 - npktServed_Flow1
                if npktArrived_Flow2 < NPKT:
                    npktArrived_Flow2 += 1
                    nextArrival_Flow2 = ArrivalTime_Flow2[npktArrived_Flow2]
                else:
                    nextArrival_Flow2 = float('inf')

        if len(queue_Flow1) > 0 and (len(queue_Flow2) == 0 or queue_Flow2[0][0] >= queue_Flow1[0][0]):
            packet = queue_Flow1.pop(0)
            if packet[0] > nextDeparture:
                now = packet[0]
            else:
                now = nextDeparture
            nextDeparture = now + packet[1]
            if npktServed_Flow1 < NPKT:
                npktServed_Flow1 += 1
                departureTime_Flow1[npktServed_Flow1] = nextDeparture

        elif len(queue_Flow2) > 0:
            packet = queue_Flow2.pop(0)
            if packet[0] > nextDeparture:
                now = packet[0]
            else:
                now = nextDeparture
            nextDeparture = now + packet[1]
            if npktServed_Flow2 < NPKT:
                npktServed_Flow2 += 1
                departureTime_Flow2[npktServed_Flow2] = nextDeparture

        npktServed = npktServed_Flow1 + npktServed_Flow2
    temp = [0] * (NPKT + 2)
    for j in range(NPKT):
        temp[j + 1] = departureTime_Flow2[j + 1] - (ArrivalTime_Flow2[j + 1])
    T2[i] = statistics.mean(temp)
    N2[i] = statistics.mean(systemState_Flow2)
    i += 1

rho1 = ConstLambda / MU
rho2 = numpy.asarray(LAMBDAS) / MU
Tananlysis = ((1-rho1*(1 - rho1 - rho2)) / MU) / ((1 - rho1) * (1 - rho1 - rho2))
plt.plot(LAMBDAS, Tananlysis)
# mylist = [number for number in T2 if number < 10]
plt.plot(LAMBDAS, T2)
plt.xlabel('ρ')
plt.ylabel('delay')
plt.title('Delay2 - Rho')
plt.show()

Nananlysis = ((1-rho1*(1 - rho1 - rho2))*rho2) / ((1 - rho1) * (1 - rho1 - rho2))
plt.plot(LAMBDAS, Nananlysis)
plt.plot(LAMBDAS, N2)
plt.xlabel('ρ')
plt.ylabel('N')
plt.title('N2 - Rho')
plt.show()
