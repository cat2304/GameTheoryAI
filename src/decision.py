def decide_tile_to_play(hand_tiles, log):
    if not hand_tiles:
        log("[决策] 没有可打出的牌")
        return None
    chosen = hand_tiles[0]
    log(f"[决策] 本轮选择出牌: {chosen}")
    return chosen