###Instagram
'''
from instapy import InstaPy
from instapy import smart_run

my_username = 'itc.emcristo'
my_password = 'JesusPoderoso'

session = InstaPy(username=my_username, password=my_password, headless_browser=False)

with smart_run(session):
    session.set_relationship_bounds(enabled=True,
                                    delimit_by_numbers=True,
                                    max_followers=5000,
                                    min_followers=20,
                                    min_following=50)

    session.follow_user_followers(['teslamotors'], amount = 10, randomize = False)
    session.end()

    session.set_dont_like(['nsfw', 'kia', 'ford'])

    session.like_by_tags(
        ['jesus', 'god', 'peace', 'holiday', 'instagood', 'fashion', 'photooftheday', 'beautiful', 'art', 'brasil',
         'car', 'happy', 'instagood', 'photography', 'bolsonaro', 'picoftheday'], amount=100)


comments = ['Nice shot! @{}',
'I love your profile! @{}',
'Your feed is an inspiration 👍',
'Just incredible 😮',
'What camera did you use @{}?',
'Love your posts @{}',
'Looks awesome @{}',
'Getting inspired by you @{}',
'🙌 Yes!',
'I can feel your passion @{} 💪']
         '''

from instapy import InstaPy
from instapy import smart_run

insta_username = 'icesurfing@live.com'
insta_password = 'Elierte2@'

comments = ['jesus']

session = InstaPy(username=insta_username,
                  password=insta_password,
                  headless_browser=False)

with smart_run(session):
    session.set_dont_include(["friend1", "friend2", "friend3"])
    session.like_by_tags(["natgeo"], amount=10)
    session.set_do_comment(enabled=True, percentage=35)
    session.set_comments(comments)
    session.join_pods(topic='sports', engagement_mode='no_comments')
