import datetime

def is_nepse_holiday(date: datetime.date) -> bool:
    """
    Checks if a given date is a holiday in Nepal where NEPSE is closed.
    Uses the python-holidays package.
    """
    try:
        import holidays
        # holidays.Nepal() returns a dict-like object mapping dates to holiday names
        np_holidays = holidays.Nepal()
        return date in np_holidays
    except ImportError:
        # Fallback if package is missing
        return False

def get_next_trading_day(current_date: datetime.date) -> datetime.date:
    """
    Given a current date, returns the next valid NEPSE trading day.
    NEPSE is closed on Fridays, Saturdays, and official Nepal holidays.
    """
    next_day = current_date + datetime.timedelta(days=1)
    
    while True:
        # Friday (4) and Saturday (5) are weekend days in Nepal
        if next_day.weekday() in (4, 5) or is_nepse_holiday(next_day):
            next_day += datetime.timedelta(days=1)
        else:
            break
            
    return next_day
