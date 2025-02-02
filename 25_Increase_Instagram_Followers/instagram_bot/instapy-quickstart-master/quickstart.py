# imports
from instapy import InstaPy
from instapy import smart_run

# login credentials
insta_username = 'itc.emcristo'
insta_password = 'JesusPoderoso'

comments = ['que show', 'paz', 'hoje', 'o', 'brasil', 'a', 'amor', 'bomdia', 'vida', 'agora', 'instagram', 'instagood', 'love', 'boanoite', 'deus', 's', 'promo', 'tbt', 'felicidade', 'today']

# get an InstaPy session!
# set headless_browser=True to run InstaPy in the background
session = InstaPy(username=insta_username,
                  password=insta_password,
                  headless_browser=False)

with smart_run(session):
    """ Activity flow """
    # general settings
    session.set_dont_include(["ednilcecorreia", "promosdaros", "otavio.santos97"])

    # activity
    session.like_by_tags(["jesus"], amount=10)

    # Joining Engagement Pods
    session.set_do_comment(enabled=True, percentage=35)
    session.set_comments(comments)
    session.join_pods(topic='sports', engagement_mode='no_comments')


