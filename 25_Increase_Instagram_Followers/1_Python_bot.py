from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from time import sleep, strftime
from random import randint
import pandas as pd
# selenium 3
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())

sleep(2)
driver.get('https://www.instagram.com/accounts/login/?source=auth_switcher')
sleep(3)

username = driver.find_element_by_name("Phone number, username, or email")
username.send_keys("cantinhodoforrolondon")
password = driver.find_element_by_name("Password")
password.send_keys("Elierte2@")

button_login = webdriver.find_element_by_css_selector('#react-root > section > main > div > article > div > div:nth-child(1) > div > form > div:nth-child(3) > button')
button_login.click()
sleep(3)

notnow = webdriver.find_element_by_css_selector('body > div:nth-child(13) > div > div > div > div.mt3GC > button.aOOlW.HoLwm')
notnow.click() #comment these last 2 lines out, if you don't get a pop up asking about notifications

############

hashtag_list = ['travelblog', 'travelblogger', 'traveler']

# prev_user_list = [] - if it's the first time you run it, use this line and comment the two below
prev_user_list = pd.read_csv('C:\\Users\\diego\\Desktop\\python\\25_Increase_Instagram_Followers\\users_followed_list.csv', delimiter=',').iloc[:,
                 1:2]  # useful to build a user log
prev_user_list = list(prev_user_list['0'])

new_followed = []
tag = -1
followed = 0
likes = 0
comments = 0

for hashtag in hashtag_list:
    tag += 1
    webdriver.get('https://www.instagram.com/explore/tags/' + hashtag_list[tag] + '/')
    sleep(5)
    first_thumbnail = webdriver.find_element_by_xpath(
        '//*[@id="react-root"]/section/main/article/div[1]/div/div/div[1]/div[1]/a/div')

    first_thumbnail.click()
    sleep(randint(1, 2))
    try:
        for x in range(1, 200):
            username = webdriver.find_element_by_xpath(
                '/html/body/div[3]/div/div[2]/div/article/header/div[2]/div[1]/div[1]/h2/a').text

            if username not in prev_user_list:
                # If we already follow, do not unfollow
                if webdriver.find_element_by_xpath(
                        '/html/body/div[3]/div/div[2]/div/article/header/div[2]/div[1]/div[2]/button').text == 'Follow':

                    webdriver.find_element_by_xpath(
                        '/html/body/div[3]/div/div[2]/div/article/header/div[2]/div[1]/div[2]/button').click()

                    new_followed.append(username)
                    followed += 1

                    # Liking the picture
                    button_like = webdriver.find_element_by_xpath(
                        '/html/body/div[3]/div/div[2]/div/article/div[2]/section[1]/span[1]/button/span')

                    button_like.click()
                    likes += 1
                    sleep(randint(18, 25))

                    # Comments and tracker
                    comm_prob = randint(1, 10)
                    print('{}_{}: {}'.format(hashtag, x, comm_prob))
                    if comm_prob > 7:
                        comments += 1
                        webdriver.find_element_by_xpath(
                            '/html/body/div[3]/div/div[2]/div/article/div[2]/section[1]/span[2]/button/span').click()
                        comment_box = webdriver.find_element_by_xpath(
                            '/html/body/div[3]/div/div[2]/div/article/div[2]/section[3]/div/form/textarea')

                        if (comm_prob < 7):
                            comment_box.send_keys('Really cool!')
                            sleep(1)
                        elif (comm_prob > 6) and (comm_prob < 9):
                            comment_box.send_keys('Nice work :)')
                            sleep(1)
                        elif comm_prob == 9:
                            comment_box.send_keys('Nice gallery!!')
                            sleep(1)
                        elif comm_prob == 10:
                            comment_box.send_keys('So cool! :)')
                            sleep(1)
                        # Enter to post comment
                        comment_box.send_keys(Keys.ENTER)
                        sleep(randint(22, 28))

                # Next picture
                webdriver.find_element_by_link_text('Next').click()
                sleep(randint(25, 29))
            else:
                webdriver.find_element_by_link_text('Next').click()
                sleep(randint(20, 26))
    # some hashtag stops refreshing photos (it may happen sometimes), it continues to the next
    except:
        continue

for n in range(0, len(new_followed)):
    prev_user_list.append(new_followed[n])

updated_user_df = pd.DataFrame(prev_user_list)
updated_user_df.to_csv('{}_users_followed_list.csv'.format(strftime("%Y%m%d-%H%M%S")))
print('Liked {} photos.'.format(likes))
print('Commented {} photos.'.format(comments))
print('Followed {} new people.'.format(followed))