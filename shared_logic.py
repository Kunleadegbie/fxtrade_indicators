# ================================
# SHARED DECISION ENGINE
# ================================

def normalize_signal(text):
    if text is None:
        return "NEUTRAL"

    if "BUY" in text:
        return "BUY"
    elif "SELL" in text:
        return "SELL"
    else:
        return "NEUTRAL"


def unified_decision(kpi_decision, trading_signal, entry_signal):

    kpi = normalize_signal(kpi_decision)
    trade = normalize_signal(trading_signal)

    # ============================
    # FULL ALIGNMENT REQUIRED
    # ============================

    if kpi == "BUY" and trade == "BUY" and "ENTER BUY" in entry_signal:
        return "EXECUTE BUY"

    if kpi == "SELL" and trade == "SELL" and "ENTER SELL" in entry_signal:
        return "EXECUTE SELL"

    return "NO TRADE"