# Data Science Course

## Types of data:

'''
1 Numerical: is the most common data type, it has different types of numerical:
            a) Quantitive measure: heights, page load times, stock prices, etc.
            b) Discrete Data: couts some event: How many purchase a customer make in a year?
            How mnay times did I flip "heads"?
            c) Continuous Data: has an infinite number of possible values. Ex: 0.1 degress, 1000 drops, 54.2 cm etc
           
2 Categorical: has no mumerical meaning, ex: gender, race, state of residence, product categorical
3 Oridnal: mixe of numerical with categorical, ex: movies rates where 1 star is less than 5 stars. Measure of quality. 

Work with some mathematical techniques related to data science:
1 Mean
2 Mediam 
3 Mode


'''

## Whitespace Is Important:
listOfNumbers = [1, 2, 3, 4, 5, 6, 7]

for number in listOfNumbers:
    print(number)
    if (number % 2 == 0):
        print("is even")
    else:
        print("is odd")
        
print ("All done.")

## Importing Modules:
import numpy as np

A = np.random.normal(25.0, 5.0, 10)
print (A)

## Lists:

# How many elements are in the list:
x = [1, 2, 3, 4, 5, 6]
print(len(x))

# Select the first 3 elements:
x[:3]

# Select the last 3 elements: 
x[3:]

# Select the last 3 elements of the list:
x[-2:]

# Append a new values to the list:
x.extend([7,8])
x
x.append(9)
x

# This build a lis of lists:
y = [10, 11, 12] # create a new variable
listOfLists = [x, y] # add to X
listOfLists # print the results

# Select the elemeant 1 from y, so 0=10, 1=11, 12=2
y[1]

z = [3, 2, 1]
z.sort()
z

z.sort(reverse=True)
z

## Tuples, they are imutables, cannot be changed:

#Tuples are just immutable lists. Use () instead of []
x = (1, 2, 3)
len(x)

y = (4, 5, 6)
y[2]

listOfTuples = [x, y]
listOfTuples

(age, income) = "32,120000".split(',')
print(age)
print(income)

## Dictionaries
# Like a map or hash table in other languages
captains = {}
captains["Enterprise"] = "Kirk"
captains["Enterprise D"] = "Picard"
captains["Deep Space Nine"] = "Sisko"
captains["Voyager"] = "Janeway"

print(captains["Voyager"])

print(captains.get("Enterprise"))

print(captains.get("NX-01"))

for ship in captains:
    print(ship + ": " + captains[ship])

## Functions
def SquareIt(x):
    return x * x

print(SquareIt(2))

#You can pass functions around as parameters
def DoSomething(f, x):
    return f(x)

print(DoSomething(SquareIt, 3))

#Lambda functions let you inline simple functions
print(DoSomething(lambda x: x * x * x, 3))

## Boolean Expressions
print(1 == 3)
print(True or False)
print(1 is 3)

# Check is 1 is actually 3:
if 1 is 3:
    print("How did that happen?")
elif 1 > 3:
    print("Yikes")
else:
    print("All is well with the world")

## Looping:
for x in range(10):
    print(x)

####
for x in range(10):
    if (x is 1):
        continue
    if (x > 5):
        break
    print(x)

#####
x = 0
while (x < 10):
    print(x)
    x += 1

## Activity, Write some code that creates a list of integers, loops through each element of the list, and only prints out even numbers!
myList = [0, 1, 2, 5, 8, 3]
for number in myList:
    if number %2 is 1: #The % is called the modulo operator. Of course when the remainder is 0, the number is even.
        print(number)

## Another example with odds number and a break on 237, remember the number are not in the right sequence:
numbers = [
    951, 402, 984, 651, 360, 69, 408, 319, 601, 485, 980, 507, 725, 547, 544, 
    615, 83, 165, 141, 501, 263, 617, 865, 575, 219, 390, 984, 592, 236, 105, 942, 941, 
    386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345, 
    399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217, 
    815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717, 
    958, 609, 842, 451, 688, 753, 854, 685, 93, 857, 440, 380, 126, 721, 328, 753, 470, 
    743, 527
]

for x in numbers:
    if x % 2 == 1: #The % is called the modulo operator. Of course when the remainder is 1, the number is odd.
        print x
    if x == 237:
        break


