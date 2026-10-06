# the file is downloaded from https://github.com/NNargesNN/GeoToken/blob/main/s2_token_utils.py

import s2sphere

def coords_to_s2_tokens(lat: float, lon: float, max_level: int = 20) -> list:
    """
    Convert (latitude, longitude) to a sequence of 21 hierarchical S2 tokens.
    - Token 0: Cube face index (0 to 5)
    - Tokens 1-20: Sub-cell quadrant position (0 to 3) for levels 1 to 20
    """
    lat_lng = s2sphere.LatLng.from_degrees(lat, lon)
    cell_id = s2sphere.CellId.from_lat_lng(lat_lng)

    # Token 0 is the root cube face
    tokens = [cell_id.face()]

    # FIX: shift must be 61 - 2*level, not 60 - 2*level.
    # A 64-bit S2 CellId is [3 bits face][61 bits position+sentinel].
    # Using 60 instead of 61 reads every level's bits one position too low,
    # which desyncs the trailing sentinel bit and produces an invalid CellId
    # (verified: s2sphere reports level=15 instead of 20, and round-tripping
    # breaks badly near S2 face boundaries, e.g. near (0,0)).
    for level in range(1, max_level + 1):
        shift = 61 - 2 * level
        child_pos = (cell_id.id() >> shift) & 3
        tokens.append(child_pos)

    return tokens


def s2_tokens_to_coords(tokens: list, max_level: int = 20) -> tuple:
    """
    Reconstruct geographic coordinates (latitude, longitude)
    from a sequence of 21 S2 tokens by building the 64-bit CellId.
    """
    if len(tokens) != max_level + 1:
        raise ValueError(f"Expected {max_level + 1} tokens, but received {len(tokens)}")

    face = tokens[0]

    # 1. Place the face bits in the top positions (bits 61..63)
    cell_id_val = face << 61

    # 2. Place each 2-bit quadrant into its exact bit position (FIX: 61 - 2*level)
    for level, child_pos in enumerate(tokens[1:], start=1):
        shift = 61 - 2 * level
        cell_id_val |= (child_pos << shift)

    # 3. Set the trailing sentinel bit that marks this CellId's level.
    #    FIX: for level L, the sentinel sits at bit (61 - 2*L) - 1.
    lsb_shift = 61 - 2 * max_level
    cell_id_val |= (1 << (lsb_shift - 1))

    cell = s2sphere.CellId(cell_id_val)
    lat_lng = cell.to_lat_lng()
    return lat_lng.lat().degrees, lat_lng.lng().degrees


if __name__ == "__main__":
    sample_lat, sample_lon = 44.8167, 20.4600

    print("=" * 65)
    print(f"[INPUT] Coordinates: Latitude = {sample_lat}, Longitude = {sample_lon} (Belgrade)")
    print("=" * 65)

    s2_tokens = coords_to_s2_tokens(sample_lat, sample_lon)
    print(f"\n[1] Generated S2 Token Sequence (Length: {len(s2_tokens)}):")
    print(f"    {s2_tokens}")
    print(f"    - Token 0 (Cube Face): {s2_tokens[0]}")
    print(f"    - Coarse Level Tokens (1-5): {s2_tokens[1:6]}")
    print(f"    - Fine Level Tokens (16-20): {s2_tokens[16:21]}")

    recon_lat, recon_lon = s2_tokens_to_coords(s2_tokens)
    print("\n" + "=" * 65)
    print(f"[2] Reconstructed Coordinates: Latitude = {recon_lat:.5f}, Longitude = {recon_lon:.5f}")

    delta_lat = abs(sample_lat - recon_lat)
    delta_lon = abs(sample_lon - recon_lon)
    print(f"[3] Discretization Error (Level 20 cell precision):")
    print(f"    - Delta Lat: {delta_lat:.8f}°")
    print(f"    - Delta Lon: {delta_lon:.8f}° (~few meters)")

    # Sanity check: the reconstructed CellId should be a *valid* level-20 cell.
    lat_lng = s2sphere.LatLng.from_degrees(sample_lat, sample_lon)
    face = s2sphere.CellId.from_lat_lng(lat_lng).face()
    cell_id_val = face << 61
    for level, child_pos in enumerate(s2_tokens[1:], start=1):
        cell_id_val |= (child_pos << (61 - 2 * level))
    cell_id_val |= (1 << ((61 - 2 * 20) - 1))
    check_cell = s2sphere.CellId(cell_id_val)
    print(f"[4] Sanity check: is_valid={check_cell.is_valid()}, level={check_cell.level()} (want True, 20)")
    print("=" * 65)