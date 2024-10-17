from datetime import timedelta, datetime

# In Europe, daylight savings is observed between the LAST Sunday of March and the LAST Sunday of October
# In the United States, daylight savings is observed between the SECOND Sunday of March and the FIRST Sunday of November

def find_last_sunday(date):
    days_left = 6 - date.weekday()
    # .weekday() - datetime.date class function that returns an integer value that corresponds to the day of the week
    # I.e. 0 - Monday, 1 - Tuesday, 2 - Wednesday, 3 - Thursday, 4 - Friday, 5 - Saturday, and 6 - Sunday
    if days_left != 0:
        date += timedelta(days = days_left) # Add number of days to obtain LAST SUNDAY of March & October
    return date 

# Defines the months of Daylight Saving Time in Europe (apart from Iceland, Belarus, Turkey, and Russia) for a specific year
# European Summer Time begins at 01:00 UTC on the LAST Sunday of March (between 25 and 31 of March) and ends at 01:00 UTC on the LAST Sunday of October (between 25 and 31 of October)
def europe_dst_range(year):
    DST_start = datetime(year = 1, month = 3, day = 25, hour = 1, minute = 0)
    DST_end = datetime(year = 1, month = 10, day = 25, hour = 1, minute = 0)

    dst_start = find_last_sunday(DST_start.replace(year = year)) # Find explicit dates of DST for specific input year
    dst_end = find_last_sunday(DST_end.replace(year = year))

    return dst_start, dst_end # Defines datetime limits for .json file corrections per year (from 2004 to 2017 - the time period we have Cassini data for)

