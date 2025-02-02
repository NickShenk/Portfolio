def calculate_indicators(prices):
    # Keep only the last 50 prices
    prevprices = prices[-51:-1]
    prices = prices[-50:]
    
    
    # Calculate Moving Averages
    def calculate_moving_average(data, window):
        return sum(data[-window:]) / window

    ma10 = calculate_moving_average(prices, 10)
    ma50 = calculate_moving_average(prices, 50)
    prevma10 = calculate_moving_average(prevprices, 10) 
    prevma50 = calculate_moving_average(prevprices, 50)
    # Calculate RSI
    def calculate_rsi(prices, window=14):
        if len(prices) < window + 1:
            return None
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = sum(delta for delta in deltas if delta > 0)
        losses = -sum(delta for delta in deltas if delta < 0)
        avg_gain = gains / window
        avg_loss = losses / window
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(prices)
    prevrsi = calculate_rsi(prevprices)
    # Calculate MACD
    # Calculate EMA
    def calculate_ema(data, window):
        if len(data) < window:
            return None
        ema_values = []
        multiplier = 2 / (window + 1)
        ema = data[0]  # Start EMA with the first data point
        ema_values.append(ema)
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
            ema_values.append(ema)
        return ema_values

    # Calculate MACD
    def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
        if len(prices) < long_window:
            return None, None
        short_ema = calculate_ema(prices, short_window)
        long_ema = calculate_ema(prices, long_window)
        macd_values = [short_ema[i] - long_ema[i] for i in range(len(short_ema))]
        signal_line = calculate_ema(macd_values, signal_window)
        return macd_values[-1], signal_line[-1]
    macd, macd_signal = calculate_macd(prices)
    prevmacd, prevmacd_signal = calculate_macd(prevprices)
    return [(ma10 - ma50)/ma10, rsi,macd - macd_signal,(prevma10 - prevma50)/prevma10, prevrsi, prevmacd - prevmacd_signal] # Divide previouse by ma10 so its numbers are more comparable to the current