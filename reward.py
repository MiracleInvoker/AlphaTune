def net_calmar_ratio(insample):
    """
    Net Calmar Ratio
    Score = (Returns - (Turnover * 252 * Transaction Cost)) / Drawdown

    5 basis points assumed for Transaction Cost
    """

    score = (insample["returns"] - (insample["turnover"] * 252 * 0.0005)) / insample["drawdown"]

    return score


def sortino_ratio(pnl, turnover):
    pass