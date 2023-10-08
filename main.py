import csv
import random
import numpy as np
import matplotlib.pyplot as plt


def castToFloats(row, dimension):
    castedRow = []
    for i in range(dimension):
        castedRow.append(float(row[i]))

    return castedRow


# csv file name
filename = "data1.csv"

points = []
maxPoints = []
minPoint = []
pointDimension = 2
m = 1.3
numberOfPoints = 1

# reading csv file
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    firstRow = next(csvreader)
    castedRow = castToFloats(firstRow, pointDimension)
    points.append(np.array(castedRow).reshape(1, pointDimension))
    pointDimension = len(castedRow)
    print(pointDimension)
    for i in range(pointDimension):
        maxPoints.append(castedRow[i])
        minPoint.append(castedRow[i])

    maxPoints = np.array(maxPoints).reshape(1, pointDimension)
    minPoint = np.array(minPoint).reshape(1, pointDimension)

    # extracting each data row one by one
    for row in csvreader:
        if len(row) > 0:
            numberOfPoints += 1
            castedRow = castToFloats(row, pointDimension)
            points.append(np.array(castedRow).reshape(1, pointDimension))

            for i in range(pointDimension):
                if castedRow[i] > maxPoints[0][i]:
                    maxPoints[0][i] = castedRow[i]
                if castedRow[i] < minPoint[0][i]:
                    minPoint[0][i] = castedRow[i]

    # print("Total no. of points: %d" % (csvreader.line_num))

V = []
centersCount = 4
U = [[float(0) for x in range(numberOfPoints)] for y in range(centersCount)]
colors = []
for i in range(centersCount):
    color = (random.random(), random.random(), random.random())
    colors.append(color)
print(len(colors))
# colors = np.array(colors).reshape(3, centersCount)

for k in range(centersCount):
    newCenters = []
    for j in range(pointDimension):
        newCenters.append(random.uniform(minPoint[0][j], maxPoints[0][j]))
    V.append(np.array(newCenters).reshape(1, pointDimension))
# print(V)

for t in range(50):
    for i in range(centersCount):
        for k in range(numberOfPoints):
            totalDists = 0
            for j in range(centersCount):
                totalDists += pow(np.linalg.norm(points[k] - V[i]) / np.linalg.norm(points[k] - V[j]), 2 / (m - 1))
            U[i][k] = 1 / totalDists
    # print(U)
    for i in range(centersCount):
        numerator = 0
        denominator = 0
        for k in range(numberOfPoints):
            numerator += pow(U[i][k], m) * points[k]
            denominator += pow(U[i][k], m)
        V[i] = numerator / denominator

    # print(V)

cost = 0
for j in range(numberOfPoints):
    for i in range(centersCount):
        cost += pow(U[i][j], m) * pow(np.linalg.norm(points[j] - V[i]), 2)
print(cost)

crispCenters = []
for k in range(numberOfPoints):
    nearestCenter = 0
    for i in range(1, centersCount):
        if U[nearestCenter][k] < U[i][k]:
            nearestCenter = i

    crispCenters.append(nearestCenter)
# crispCenters = np.array(crispCenters).reshape(1,numberOfPoints)

for k in range(numberOfPoints):
    plt.scatter(points[k][0][0], points[k][0][1], color=colors[crispCenters[k]], s=6)

print(V)
for i in range(centersCount):
    plt.scatter(V[i][0][0], V[i][0][1], c='black', marker=r'$\odot$', facecolors='none', edgecolors='black')
plt.show()
