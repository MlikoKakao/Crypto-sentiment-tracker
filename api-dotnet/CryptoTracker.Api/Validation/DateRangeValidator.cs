namespace CryptoTracker.Api.Validation;

public static class DateRangeValidator
{
    public static bool IsValid(DateTimeOffset startDate, DateTimeOffset endDate)
    {
        return endDate >= startDate;
    }
}
