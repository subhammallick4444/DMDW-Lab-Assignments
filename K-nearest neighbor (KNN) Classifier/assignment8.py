import math

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)
def knn_classify(data, labels, new_point, k):
    distances = []
    for i in range(len(data)):
        dist = euclidean_distance(data[i], new_point)
        distances.append((dist, labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    class_counts = {}
    for neighbor in neighbors:
        _, label = neighbor
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    max_count = 0
    predicted_class = None
    for label, count in class_counts.items():
        if count > max_count:
            max_count = count
            predicted_class = label
    
    return predicted_class
data = [
    [1.7, 65, 20],
    [1.9, 85, 33],
    [1.78, 76, 31],
    [1.73, 74, 24],
    [1.81, 75, 35],
    [1.73, 70, 75],
    [1.8, 71, 63],
    [1.75, 69, 25]
]

labels = [
    "Programmer",
    "Builder",
    "Builder",
    "Programmer",
    "Builder",
    "Scientist",
    "Scientist",
    "Programmer"
]

new_point = [1.69, 79, 37]
k = 3  
predicted_class = knn_classify(data, labels, new_point, k)
print("Predicted Class:", predicted_class)
