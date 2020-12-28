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
    ArrivalTime_Flow1 = numpy.random.exponential(scale=1 / lam, size=(NPKT + 1))
    numpy.random.shuffle(ArrivalTime_Flow1)
    ArrivalTime_Flow1 = numpy.cumsum(ArrivalTime_Flow1)
    ArrivalTime_Flow2 = numpy.random.exponential(scale=1 / ConstLambda, size=(NPKT + 1))
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
    queue_Flow1 = []
    queue_Flow2 = []

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
            if len(queue_Flow1) > 0 and (queue_Flow1[0][0] < nextDeparture):
                remainServiceTime_Flow2 = nextDeparture - queue_Flow1[0][0]
                queue_Flow2.insert(0, (queue_Flow1[0][0], remainServiceTime_Flow2))
            elif npktServed_Flow2 < NPKT:
                npktServed_Flow2 += 1
                departureTime_Flow2[npktServed_Flow2] = nextDeparture

        npktServed = npktServed_Flow1 + npktServed_Flow2
    temp = [0] * (NPKT + 2)
    for j in range(NPKT):
        temp[j + 1] = departureTime_Flow1[j + 1] - ArrivalTime_Flow1[j + 1]
    T1[i] = statistics.mean(temp)
    N1[i] = statistics.mean(systemState_Flow1)
    i += 1

rho2 = ConstLambda / MU
rho1 = numpy.asarray(LAMBDAS) / MU
Tananlysis = (1.0 / MU) / (1 - rho1)
plt.plot(LAMBDAS, Tananlysis)
plt.plot(LAMBDAS, T1)
plt.xlabel('ρ')
plt.ylabel('delay')
plt.title('Delay - Rho')
plt.show()

Nananlysis = (rho1) / (1 - rho1)
plt.plot(LAMBDAS, Nananlysis)
plt.plot(LAMBDAS, N1)
plt.xlabel('ρ')
plt.ylabel('N')
plt.title('N - Rho')
plt.show()
