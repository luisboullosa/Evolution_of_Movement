# Central place for layout coordinates and sizes used by visualize_individual
# Tweak these values to adjust figure layout without editing the main script

# figure gridspec ratios
GS_WIDTH_RATIOS: list[float] = [1.0, 0.35]
GS_HEIGHT_RATIOS = [1.2, 1, 1.2]

# info box position (figure coordinates)
AX_INFO_POS: list[float] = [0.74, 0.68, 0.22, 0.22]
AX_INFO_ALPHA = 0.95

# slider position
AX_SLIDER_POS: list[float] = [0.2, 0.02, 0.6, 0.03]
AX_SLIDER_FACECOLOR = "0.9"

# play button and resp toggle positions
AX_PLAY_POS: list[float] = [0.82, 0.92, 0.08, 0.04]
AX_RESP_TOGGLE_POS: list[float] = [0.74, 0.92, 0.06, 0.04]

# prev/next/random/text box positions
AX_TEXT_POS: list[float] = [0.01, 0.05, 0.12, 0.05]
AX_PREV_POS: list[float] = [0.01, 0.82, 0.045, 0.05]
AX_NEXT_POS: list[float] = [0.06, 0.82, 0.045, 0.05]
AX_RAND_POS: list[float] = [0.11, 0.82, 0.045, 0.05]

# NN axis area (for potential adjustments)
AX_NN_GRIDPOS = (1, 0)  # same as in the main gridspec usage
