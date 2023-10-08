import csv
import random
import numpy as np
import matplotlib.pyplot as plt


class FCM:
    def __init__(self, filename, m):
        self.filename = filename
        self.m = m

        self.points = []
        self.V = []
        self.U = []
        self.pointDimension = 0
        self.numberOfPoints = 0
        self.centersCount = 0

        self.colors = []
        self.maxPoints = []
        self.minPoint = []
        self.crispCenters = []

        self.readFile()

    def castToFloats(self, point, dimension):
        castedRow = []
        for i in range(dimension):
            castedRow.append(float(point[i]))

        return castedRow

    def readFile(self):
        maxPoints = []
        minPoints = []
        with open(self.filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            firstRow = next(csvreader)
            self.pointDimension = len(firstRow)
            castedRow = self.castToFloats(firstRow, self.pointDimension)
            self.points.append(np.array(castedRow).reshape(1, self.pointDimension))
            for i in range(self.pointDimension):
                maxPoints.append(castedRow[i])
                minPoints.append(castedRow[i])

            self.maxPoints = np.array(maxPoints).reshape(1, self.pointDimension)
            self.minPoint = np.array(minPoints).reshape(1, self.pointDimension)

            for row in csvreader:
                if len(row) > 0:
                    self.numberOfPoints += 1
                    castedRow = self.castToFloats(row, self.pointDimension)
                    self.points.append(np.array(castedRow).reshape(1, self.pointDimension))

                    for i in range(self.pointDimension):
                        if castedRow[i] > self.maxPoints[0][i]:
                            self.maxPoints[0][i] = castedRow[i]
                        if castedRow[i] < self.minPoint[0][i]:
                            self.minPoint[0][i] = castedRow[i]

    def createRandomColors(self):
        for i in range(self.centersCount):
            color = (random.random(), random.random(), random.random())
            self.colors.append(color)

    def createRandomVertics(self, numberOfVertices):
        self.centersCount = numberOfVertices
        self.U = [[float(0) for x in range(self.numberOfPoints)] for y in range(self.centersCount)]

        for k in range(self.centersCount):
            newCenters = []
            for j in range(self.pointDimension):
                newCenters.append(random.uniform(self.minPoint[0][j], self.maxPoints[0][j]))
            self.V.append(np.array(newCenters).reshape(1, self.pointDimension))

    def learn(self, numberOfVertices):
        self.createRandomVertics(numberOfVertices)
        for t in range(50):
            for i in range(self.centersCount):
                for k in range(self.numberOfPoints):
                    totalDists = 0
                    for j in range(self.centersCount):
                        totalDists += pow(
                            np.linalg.norm(self.points[k] - self.V[i]) / np.linalg.norm(self.points[k] - self.V[j]),
                            2 / (self.m - 1))
                    self.U[i][k] = 1 / totalDists
            # print(U)
            for i in range(self.centersCount):
                numerator = 0
                denominator = 0
                for k in range(self.numberOfPoints):
                    numerator += pow(self.U[i][k], self.m) * self.points[k]
                    denominator += pow(self.U[i][k], self.m)
                self.V[i] = numerator / denominator

        return self.getCost()

    def convertFuzzyToCrisp(self):
        for k in range(self.numberOfPoints):
            nearestCenter = 0
            for i in range(1, self.centersCount):
                if self.U[nearestCenter][k] < self.U[i][k]:
                    nearestCenter = i

            self.crispCenters.append(nearestCenter)

    def plotTwoDimensionalPoints(self):
        self.createRandomColors()
        self.convertFuzzyToCrisp()

        for k in range(self.numberOfPoints):
            plt.scatter(self.points[k][0][0], self.points[k][0][1], color=self.colors[self.crispCenters[k]], s=6)

        print(self.V)
        for i in range(self.centersCount):
            plt.scatter(self.V[i][0][0], self.V[i][0][1], c='black', marker=r'$\odot$', facecolors='none',
                        edgecolors='black')
        plt.show()

    def getCost(self):
        cost = 0
        for j in range(self.numberOfPoints):
            for i in range(self.centersCount):
                cost += pow(self.U[i][j], self.m) * pow(np.linalg.norm(self.points[j] - self.V[i]), 2)

        return cost

    def plot_costs(self, costs):
        plt.plot(costs)
        plt.ylabel('Cost')
        plt.xlabel("Verticals")
        plt.show()


if __name__ == '__main__':
    filename = "data1.csv"
    fcm = FCM(filename, m=1.3)

    ######### Costs for different number of centroids
    costs = []
    for i in range(2, 14):
        costs.append(fcm.learn(i))

    fcm.plot_costs(costs)
    ##################################################

    ######## Plot points in two deminesion
    # fcm.learn(4)
    # fcm.plotTwoDimensionalPoints()
    ##################################################

