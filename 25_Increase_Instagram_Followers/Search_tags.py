
from instapy import InstaPy

my_username = 'itc.emcristo'
my_password = 'JesusPoderoso'

session = InstaPy(username = my_username, password = my_password, headless_browser = True)
session.login()

session.set_relationship_bounds(enabled = True, max_followers = 200)

session.set_do_follow(True, percentage=100)


session.like_by_tags(["jesus", "igreja","paz"], amount = 3)
session.set_dont_like(["non"])

#session.unfollow_users(amount=6, allFollowing=True, sleep_delay=60)

session.end()