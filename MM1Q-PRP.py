import matplotlib.pyplot as plt
import numpy
import statistics

class MM1_PRP:

    def servingPackets_Flow1(self):
        global serverIsBusy_Flow1, serverIsBusy_Flow2, nextArrival_Flow1, now, queueLength_Flow1, npktArrived_Flow1, \
            npktServed_Flow1, nextDeparture, systemState_Flow2, npktArrived_Flow2, queueLength_Flow2
        serverIsBusy_Flow1 = True
        serverIsBusy_Flow2 = False
        systemState_Flow1[npktArrived_Flow1] = queueLength_Flow1
        queueLength_Flow1 += 1
        systemState_Flow2[npktArrived_Flow2] = queueLength_Flow2 + queueLength_Flow1
        if queueLength_Flow1 == 1:
            npktServed_Flow1 += 1
            nextDeparture = now + serviceTime_Flow1[npktServed_Flow1]
            departureTime_Flow1[npktServed_Flow1] = nextDeparture
            queueLength_Flow1 -= 1
            systemState_Flow2[npktArrived_Flow2] -= 1
        if npktArrived_Flow1 < NPKT:
            npktArrived_Flow1 += 1
            nextArrival_Flow1 = ArrivalTime_Flow1[npktArrived_Flow1]
        else:
            nextArrival_Flow1 = float('inf')

    def servingPackets_Flow2(self):
        global serverIsBusy_Flow1, serverIsBusy_Flow2, nextArrival_Flow2, npktServed_Flow2, \
            nextDeparture, systemState_Flow2, npktArrived_Flow2, queueLength_Flow2, queueLength_Flow1
        serverIsBusy_Flow2 = True
        serverIsBusy_Flow1 = False
        queueLength_Flow2 = queueLength_Flow1 + 1
        systemState_Flow2[npktArrived_Flow2] = queueLength_Flow2
        if queueLength_Flow2 == 1:
            npktServed_Flow2 += 1
            nextDeparture = now + serviceTime_Flow2[npktServed_Flow2]
            departureTime_Flow2[npktServed_Flow2] = nextDeparture
            queueLength_Flow2 -= 1
            systemState_Flow2[npktArrived_Flow2] -= 1
        if npktArrived_Flow2 < NPKT:
            npktArrived_Flow2 += 1
            nextArrival_Flow2 = ArrivalTime_Flow2[npktArrived_Flow2]
        else:
            nextArrival_Flow2 = float('inf')


if __name__ == '__main__':

    mm1_prp = MM1_PRP()
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
        ArrivalTime_Flow1 = numpy.cumsum(ArrivalTime_Flow1)
        ArrivalTime_Flow2 = numpy.random.exponential(scale=1 / lam, size=(NPKT + 1))
        ArrivalTime_Flow2 = numpy.cumsum(ArrivalTime_Flow2)

        serviceTime_Flow1 = numpy.random.exponential(scale=1 / MU, size=(NPKT + 1))
        serviceTime_Flow2 = numpy.random.exponential(scale=1 / MU, size=(NPKT + 1))

        nextDeparture = float('inf')
        nextArrival_Flow2 = ArrivalTime_Flow2[1]
        nextArrival_Flow1 = ArrivalTime_Flow1[1]
        if nextArrival_Flow1 < nextArrival_Flow2:
            nextDeparture = nextArrival_Flow1 + serviceTime_Flow1[1]
        else:
            nextDeparture = nextArrival_Flow2 + serviceTime_Flow2[1]

        npktArrived_Flow1 = 1
        npktArrived_Flow2 = 1
        npktServed_Flow1 = 0
        npktServed_Flow2 = 0
        serverIsBusy_Flow1 = False
        serverIsBusy_Flow2 = False
        queueLength_Flow1 = 0
        queueLength_Flow2 = 0
        remainServiceTime_Flow2 = 0
        while npktServed < 2 * NPKT:
            if (nextArrival_Flow1 < nextDeparture) & serverIsBusy_Flow2:
                now = nextArrival_Flow1
                remainServiceTime_Flow2 = nextDeparture - now
                queueLength_Flow2 = queueLength_Flow1 +1
                systemState_Flow2[npktArrived_Flow2] = queueLength_Flow2
                mm1_prp.servingPackets_Flow1()

            elif (nextArrival_Flow1 < nextDeparture) & serverIsBusy_Flow1:
                now = nextDeparture
                mm1_prp.servingPackets_Flow1()

            elif (nextDeparture < nextArrival_Flow1) & (remainServiceTime_Flow2 > 0):
                serverIsBusy_Flow2 = True
                serverIsBusy_Flow1 = False
                departureTime_Flow2[npktServed_Flow2] = now + remainServiceTime_Flow2
                systemState_Flow2[npktArrived_Flow2] = queueLength_Flow2 + queueLength_Flow1
                remainServiceTime_Flow2 = 0
                npktServed_Flow2 += 1
                queueLength_Flow2 -= 1
                if npktArrived_Flow2 < NPKT:
                    npktArrived_Flow2 += 1

            elif nextArrival_Flow1 < nextArrival_Flow2:
                now = nextArrival_Flow1
                mm1_prp.servingPackets_Flow1()

            else:
                now = nextArrival_Flow2
                mm1_prp.servingPackets_Flow2()

            npktServed = npktServed_Flow1 + npktServed_Flow2
        temp = [0] * (NPKT + 2)
        for j in range(NPKT):
            temp[j] = departureTime_Flow2[j + 1] - ArrivalTime_Flow2[j + 1] - serviceTime_Flow2[j + 1]
        T2[i] = numpy.sum(temp) / 500
        N2[i] = statistics.mean(systemState_Flow2)
        i += 1

    rho1 = ConstLambda / MU
    rho2 = numpy.asarray(LAMBDAS) / MU
    Tananlysis = (1.0 / MU) / ((1 - rho1) * (1 - rho1 - rho2))
    plt.plot(LAMBDAS, Tananlysis)
    plt.plot(LAMBDAS, T2)
    plt.xlabel('ρ')
    plt.ylabel('delay')
    plt.title('Delay - Rho')
    plt.show()

    Nananlysis = (rho2) / ((1 - rho1) * (1 - rho1 - rho2))
    plt.plot(LAMBDAS, Nananlysis)
    plt.plot(LAMBDAS, N2)
    plt.xlabel('ρ')
    plt.ylabel('N')
    plt.title('N - Rho')
    plt.show()
