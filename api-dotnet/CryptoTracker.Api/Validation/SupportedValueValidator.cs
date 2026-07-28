namespace CryptoTracker.Api.Validation;

public static class SupportedValueValidator
{
    private static readonly HashSet<string> SupportedCoins =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "BTC",
            "ETH",
            "XMR"
        };

    private static readonly HashSet<string> SupportedAnalyzers =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "vader",
            "textblob",
            "twitter-roberta",
            "finbert"
        };

    private static readonly HashSet<string> SupportedSources =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "reddit",
            "youtube",
            "news"
        };

    public static bool IsSupportedCoin(string coin)
    {
        return SupportedCoins.Contains(coin);
    }

    public static bool IsSupportedAnalyzer(string analyzer)
    {
        return SupportedAnalyzers.Contains(analyzer);
    }

    public static IReadOnlyCollection<string> GetSupportedAnalyzers()
    {
        return SupportedAnalyzers;
    }

    public static bool IsSupportedSource(string source)
    {
        return SupportedSources.Contains(source);
    }
}
