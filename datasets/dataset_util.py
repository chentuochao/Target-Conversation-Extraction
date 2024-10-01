import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf


MIC_WALL_SPACING = 0.8
MIC_HEIGHT = 1.5
HEAD_SIZE = 0.2



def overlap_duration_check(dialogue):
    self_speaker = dialogue["self_speech"]
    others = dialogue["others"]
    L_total = self_speaker["speech"].shape[-1]

    timestamps = [self_speaker["timestamp"]]
    search_index = [0]
    for o in others:
        timestamps.append(o["timestamp"])
        search_index.append(0)
    
    step = 160
    # print(len(timestamps), L_total)
    # print(timestamps[0])
    overlap_duration = 0
    valid_duration = 0
    
    diag_info = {}

    duration = []
    for stamp in timestamps:
        du = 0
        for b, e in stamp:
            du += (e - b) 
        duration.append(du)
    duration = np.array(duration)
    duration = duration/np.sum(duration)
    diag_info["occupation"] = duration.tolist() 
    
    for i in range(0, L_total, step):
        begin = i
        end = i + step

        spk_exits = []

        for j in range(len(timestamps)):
            while True:
                idx = search_index[j]
                if idx >= len(timestamps[j]):
                    break
                
                b, e = timestamps[j][idx]
                if end <= b:
                    break
                elif e <= begin:
                    search_index[j] += 1
                    continue
                else:
                    spk_exits.append(j)
                    if e < end:
                        search_index[j] += 1
                    break

        if len(spk_exits) > 1:
            overlap_duration += step
        if len(spk_exits) > 0:
            valid_duration += step
    diag_info["valid_duration"] = valid_duration
    diag_info["overlap_duration"] = overlap_duration
    diag_info["overlap_ratio"] = overlap_duration/(valid_duration + 1e-5)
    return diag_info

def point_in_box(pos, left, right, top, bottom):
    return pos[0] >= left and pos[0] <= right and pos[1] <= top and pos[1] >= bottom


### generate the random positions of microphones in the room
def get_random_mic_positions(n_mics, left, right, bottom, top):
    """
    Generates an array of microphone positions satisfying requirements
    defined in is_valid_mic_array()
    """
    ### n_mics = 1
    min_x = left + MIC_WALL_SPACING
    max_x = right - MIC_WALL_SPACING
    
    min_y = bottom + MIC_WALL_SPACING
    max_y = top - MIC_WALL_SPACING


    # mic_positions = np.zeros((2, 3))

    center_pos_x = ( max_x - min_x )*np.random.random() + min_x
    center_pos_y = ( max_y - min_y )*np.random.random() + min_y
    MIC_HEIGHT_SIM = np.random.uniform(low = MIC_HEIGHT - 0.3, high = MIC_HEIGHT + 0.3)
    mic_center = np.array([[center_pos_x, center_pos_y, MIC_HEIGHT_SIM]])    

    # theta = np.random.uniform(low = - np.pi, high= np.pi)
    # theta_deg = np.rad2deg(theta)
    # ## left ear
    # mic_positions[0, :] = np.array([center_pos_x + HEAD_SIZE/2*np.cos(theta), center_pos_y + HEAD_SIZE/2*np.sin(theta),  MIC_HEIGHT_SIM] )
    # mic_positions[1, :] = np.array([center_pos_x - HEAD_SIZE/2*np.cos(theta), center_pos_y - HEAD_SIZE/2*np.sin(theta),  MIC_HEIGHT_SIM] )
    
    return mic_center #mic_positions, theta_deg



def choose_point_with_circle_keepout(left, right, down, up, center, R_min, R_max, h_mic):
    R =  np.random.uniform(low=R_min, high=R_max)
    

    ### find a rando, angle in the room space
    angle_offset = np.random.uniform(low = 0, high = 1) 
    angles = np.deg2rad(np.arange(0,360,1) + angle_offset)
    angel_index = np.arange(0,360,1)
    pos_x = R*np.cos(angles) + center[0]
    pos_y = center[1] + R*np.sin(angles)

    inside = (pos_x > left) & (pos_x < right) & (pos_y > down) & (pos_y < up)

    valid = np.sum(inside)

    if valid == 0:
        print("no radias intersection!")
        return choose_point_with_circle_keepout(left, right, up, down,  center, R_min, R_max, h_mic)
    
    angel_choice = angel_index[inside]
    a = np.random.choice(angel_choice)
    voice_x = pos_x[a]
    voice_y = pos_y[a]
    angle = np.rad2deg(angles[a])
    voice_h = np.random.uniform(low = h_mic - 0.5, high = h_mic + 0.5) ### height of voice
    return R, angle, np.array([voice_x, voice_y, voice_h])


def get_random_speaker_positions_dis_uniform(n_voices, mic_positions, left, right, up, down):
    voices = []
    angles = []
    distances = []
    mic_center = mic_positions[0]
    h_mic = mic_positions[0][-1]
    #print(mic_center)
    DESK_WALL_SAFE = 0.25
    SPEAK_MINX = left + DESK_WALL_SAFE
    SPEAK_MAXX = right - DESK_WALL_SAFE
    SPEAK_MINY = down + DESK_WALL_SAFE
    SPEAK_MAXY = up - DESK_WALL_SAFE

    r1 = np.linalg.norm([SPEAK_MINX - mic_center[0], SPEAK_MINY - mic_center[1]]) 
    r2 = np.linalg.norm([SPEAK_MAXX - mic_center[0], SPEAK_MINY - mic_center[1]]) 
    r3 = np.linalg.norm([SPEAK_MINX - mic_center[0], SPEAK_MAXY - mic_center[1]]) 
    r4 = np.linalg.norm([SPEAK_MAXX - mic_center[0], SPEAK_MAXY - mic_center[1]]) 
    R_max = max([r1,r2,r3,r4]) - 0.2
    
    ### self speech voices source
    for i in range(n_voices):
        while True:
            R, angle, pos = choose_point_with_circle_keepout(SPEAK_MINX, SPEAK_MAXX, SPEAK_MINY, SPEAK_MAXY, mic_center, 0.4, R_max - 1, h_mic)
            VALID = True
            for j, pos2 in enumerate(voices):
                angle2 = angles[j]

                if np.linalg.norm(pos2[:2] - pos[:2]) < 0.2 or np.abs(angle2 - angle) < 2:
                    # print(np.linalg.norm(pos2 - pos), max_delta)
                    # print("Retrying random voice location generation ... because source too close")
                    VALID = False
                    break
            if VALID:
                break

        voices.append(pos)
        angles.append(angle)
        distances.append(R)


    spk_info = []
    for i in range(0, len(voices)):
        dat = {
            "pos": voices[i].tolist(),
            "angle": angles[i],
            "distance": distances[i],
        }
        spk_info.append(dat)
    return voices, spk_info

### generate a virtual room in pyroomacoustics
def generate_room(n_spk, args, max_order, absorption):
    ## randomly generaet room size
    LWALL_MIN, LWALL_MAX = 0,0 #-5,-3 #-20, -15 meter -4, -3
    RWALL_MIN, RWALL_MAX = 4, 8 #3, 5 #15, 20
    BWALL_MIN, BWALL_MAX = 0,0 #-5, -3 #-20, -15
    TWALL_MIN, TWALL_MAX = 4, 6 #3, 5 #15, 20
    
    CEIL_MIN, CEIL_MAX = 3, 5

    left = np.random.uniform(low=LWALL_MIN, high=LWALL_MAX)
    right = np.random.uniform(low=RWALL_MIN, high=RWALL_MAX)
    top = np.random.uniform(low=TWALL_MIN, high=TWALL_MAX)
    bottom = np.random.uniform(low=BWALL_MIN, high=BWALL_MAX)    
    ceiling = np.random.uniform(low=CEIL_MIN, high=CEIL_MAX)

    absorption = np.random.uniform(low=absorption[0], high=absorption[1])
    max_order = np.random.randint(low=max_order[0], high=max_order[1]) #8, 72 #args.max_order

    ### generate the microphone
    mic_pos = get_random_mic_positions(1, left, right, bottom, top)
    voice_positions, spk_info = get_random_speaker_positions_dis_uniform(n_spk, mic_pos, left=left, right=right, up=top, down=bottom)
    # Verify speaker mic placement
    for pos in voice_positions:
        assert point_in_box(pos, left, right, top, bottom)
    assert point_in_box(mic_pos[0], left, right, top, bottom)
    # print(mic_pos)
    # print(voice_positions)
    room_info = {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "ceiling" : ceiling,
        "absorption": absorption,
        "max_order": max_order,
        "mic": mic_pos.tolist()
    }

    return mic_pos, voice_positions, room_info, spk_info

### start simulaton of pyroomacoustic
def generate_data_scenario(mic_positions,
                           voice_positions,
                           voices_data,
                           corners,
                           args,
                           absorption,
                           max_order,
                           ceiling=None,
                           total_samples = 16000):
    n_mic = mic_positions.shape[0]

    length = np.max(corners[0])
    width = np.max(corners[1])

    room_dims = [length, width, ceiling]
    room = pra.ShoeBox(p=room_dims,
                        fs=args.sr,
                        max_order=max_order,
                        absorption = absorption)


    ### add microphone array to the simulator 
    # print("mic_positions", mic_positions)
    room.add_microphone_array(mic_positions.T)
    # print(room_dims, voice_positions)
    for voice_idx in range(len(voice_positions)):
        voice_loc = voice_positions[voice_idx]
        # print(voice_idx, voice_loc, voices_data[voice_idx].shape)
        room.add_source(voice_loc, signal=voices_data[voice_idx][0, :])

    premix_reverb = room.simulate(return_premix=True)
    # print("premix_reverb = ", premix_reverb.shape)
    
    voices = []
    for i in range(len(voice_positions)):        
        gt = premix_reverb[i].copy()
        gt = gt[:1, :total_samples]
        if np.max(np.abs(gt)) < 1e-5:
            assert(i > 0)
        else:
            gt = gt/np.abs(gt).max()
        voices.append(gt)
    return voices



### augment with reverberation 
def augment_reverbration(voices_data, args, total_samples, max_order = [5, 30], absorption = [0.3, 0.9]):
    
    n_spk = len(voices_data) ###
    mic_positions, voice_positions, room_info, spk_info = generate_room(n_spk, args, max_order = max_order, absorption = absorption)
    left, right, bottom, top = room_info["left"], room_info["right"], room_info["bottom"], room_info["top"]
    corners = np.array([[left, bottom], [left, top],
                        [right, top], [right, bottom]]).T

    voices = generate_data_scenario(
                            mic_positions,
                            voice_positions,
                            voices_data,
                            corners,
                            args,
                            room_info["absorption"],
                            room_info["max_order"],
                            room_info["ceiling"],
                            total_samples)
    
    return voices, room_info, spk_info