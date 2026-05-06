import math

# Sun center, radius, max fleet speed
CX, CY  = 50.0, 50.0
SUN_R   = 10.0
MAX_SPD = 6.0

# Turn counter — incremented each call so orbital positions can be predicted
_turn = [0]


# ── Physics helpers ────────────────────────────────────────────────────────────

def _speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPD - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def _planet_pos(planet, abs_turn, av, init_planets):
    pid, _, px, py, pr = planet[0], planet[1], planet[2], planet[3], planet[4]
    init = next((p for p in init_planets if p[0] == pid), None)
    if init is None:
        return px, py
    ix, iy = init[2], init[3]
    r = math.hypot(ix - CX, iy - CY)
    if r + pr < 50.0:                       # inner planet orbits
        a0 = math.atan2(iy - CY, ix - CX)
        a  = a0 + av * abs_turn
        return CX + r * math.cos(a), CY + r * math.sin(a)
    return px, py                           # outer planet is static


def _intercept(fx, fy, target, ships, turn, av, init_planets, iters=8):
    spd = _speed(ships)
    tx, ty = target[2], target[3]
    for _ in range(iters):
        dist  = math.hypot(fx - tx, fy - ty)
        t_arr = round(dist / spd)
        tx, ty = _planet_pos(target, turn + t_arr, av, init_planets)
    return math.atan2(ty - fy, tx - fx), tx, ty


def _clears_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CX, y1 - CY
    a = dx * dx + dy * dy
    if a < 1e-10:
        return math.hypot(fx, fy) >= SUN_R
    b    = 2.0 * (fx * dx + fy * dy)
    c    = fx * fx + fy * fy - SUN_R * SUN_R
    disc = b * b - 4 * a * c
    if disc < 0:
        return True
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return not ((0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1))


# ── Threat detection ───────────────────────────────────────────────────────────

def _threats(planets, fleets, player):
    result = {p[0]: 0 for p in planets if p[1] == player}
    for fleet in fleets:
        fid, fown, fx, fy, fang, _, fships = fleet
        if fown == player:
            continue
        for planet in planets:
            if planet[1] != player:
                continue
            pid, px, py = planet[0], planet[2], planet[3]
            to_ang = math.atan2(py - fy, px - fx)
            diff   = abs(math.atan2(math.sin(to_ang - fang), math.cos(to_ang - fang)))
            if diff < 0.15 and math.hypot(fx - px, fy - py) < 60.0:
                result[pid] = result.get(pid, 0) + fships
    return result


# ── Main agent ─────────────────────────────────────────────────────────────────

def agent(obs):
    moves = []
    turn  = _turn[0]
    _turn[0] += 1

    if isinstance(obs, dict):
        player  = obs.get('player', 0)
        planets = list(obs.get('planets', []))
        fleets  = list(obs.get('fleets', []))
        av      = obs.get('angular_velocity', 0.035)
        init_p  = list(obs.get('initial_planets', []))
    else:
        player  = obs.player
        planets = [list(p) for p in obs.planets]
        fleets  = [list(f) for f in obs.fleets]
        av      = obs.angular_velocity
        init_p  = [list(p) for p in obs.initial_planets]

    my_planets = [p for p in planets if p[1] == player]
    enemy_ps   = [p for p in planets if p[1] != player]

    if not enemy_ps or not my_planets:
        return moves

    threat = _threats(planets, fleets, player)

    for mine in my_planets:
        pid, _, mx, my_c, mr, garr, _ = mine
        incoming = threat.get(pid, 0)
        reserve  = (incoming + 5) if incoming > 0 else 5
        sendable = garr - reserve

        if sendable < 1:
            continue

        best = None
        for tgt in enemy_ps:
            tid, town, tx, ty, tr, t_ships, t_prod = tgt

            spd  = _speed(sendable)
            dist = math.hypot(mx - tx, my_c - ty)
            eta  = dist / spd

            # Enemy planets produce ships during travel; neutral planets don't
            if town >= 0:
                ships_at_arr = t_ships + int(t_prod * eta)
            else:
                ships_at_arr = t_ships

            needed = ships_at_arr + 1
            if needed > sendable:
                continue

            angle, ptx, pty = _intercept(mx, my_c, tgt, needed, turn, av, init_p)

            if not _clears_sun(mx, my_c, ptx, pty):
                continue

            # Prefer high-production planets that are close
            score = (t_prod + 1) / (eta + 1.0)
            if best is None or score > best[0]:
                best = (score, angle, needed)

        if best:
            _, angle, ships = best
            moves.append([pid, angle, ships])

    return moves
