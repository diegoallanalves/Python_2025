'''
We can create the following simple function to apply Bayes’ Theorem in Python:

For example, suppose the probability of the weather being cloudy is 40%.

Also suppose the probability of rain on a given day is 20%.

Also suppose the probability of clouds on a rainy day is 85%.

If it’s cloudy outside on a given day, what is the probability that it will rain that day?

Solution:
- For example, suppose the probability of the weather being cloudy is 40%.
P(cloudy) = 0.40
- Also suppose the probability of rain on a given day is 20%.
P(rain) = 0.20
- Also suppose the probability of clouds on a rainy day is 85%.
P(cloudy | rain) = 0.85

Thus, we can calculate:
P(rain | cloudy) = P(rain) * P(cloudy | rain) / P(cloudy)
P(rain | cloudy) = 0.20 * 0.85 / 0.40
P(rain | cloudy) = 0.425

If it’s cloudy outside on a given day, the probability that it will rain that day is 42.5%.

We can create the following simple function to apply Bayes’ Theorem in Python:
'''
#define function for Bayes' theorem:

def bayesTheorem(pA, pB, pBA):
    return pA * pBA / pB

#define probabilities
pRain = 0.2
pCloudy = 0.4
# Most important part: the historic statistic
pCloudyRain = 0.85

#use function to calculate conditional probability
print(bayesTheorem(pRain, pCloudy, pCloudyRain)*100,"%")



