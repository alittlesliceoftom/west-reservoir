import sys
from datetime import datetime
from meteostat import Point, Monthly, Daily

# Set time period
start = datetime(2000, 1, 1)
end = datetime.today()

location = Point(51.4545, 2.5879)
# location = Point(49.2497, -123.1193, 70)

data = Daily(location, start, end)
data = data.fetch()


with open('bristol_daily.p','wb') as f:
    data.to_pickle(f)
