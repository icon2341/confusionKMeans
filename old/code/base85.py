# Convert to or from Z85 encoding
# Z85 Spec: https://rfc.zeromq.org/spec/32/
# 32bit (time,val) and 16bit [(time,val),(time,val)] both supported
# Skye Rhomberg, Camille Mince
# 6/9/21
# Last Updated: 7/1/21

import math
import numpy as np

########################################################################################
# Helpers

# String encoding for each base 85 digit
key = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"

# Binary representation of x in n bits
get_bin = lambda x, n: format(x, 'b').zfill(n)

########################################################################################
# 32-bit Binary Integer <-> 5-byte z85 String Conversion

def bin_to_z85(num):
    '''
    Convert a 32-bit int to z85
    '''
    return "".join([key[num//(85**i)%85] for i in range(5)])[::-1]

def z85_to_bin(ltrs):
    '''
    Convert 5 z85 letters to a 32-bit int
    '''
    return sum([85**i * key.index(l) for i,l in enumerate(ltrs[::-1])])

def z85_encode(nums):
    '''
    Encode input as z85 string (must be rounded to 32 bits)
    Input: array of 32bit integers
    '''
    return "".join([bin_to_z85(n) for n in nums])

def z85_decode(string):
    '''
    Decode z85 to list of integers
    '''
    # Each z85 int is exactly 5 chars <-> 32bit binary int
    chunks = [string[i:i+5] for i in range(0,len(string),5)]
    return [z85_to_bin(l) for l in chunks]

########################################################################################
# 32-bit format: 5char z85 = (22bit time,10bit val)

def time_encode(times):
    '''
    Encode list of (time,val) pairs as z85
    First Element is Start Time
    '''
    # (time,val) -> timeval as 32bit int -> z85
    # First element processed separately as ms since epoch
    return z85_encode([times[0]] + [1024 * t[0] + t[1] for t in times[1:]]);

def time_decode(string):
    '''
    Decode z85 to time-series
    Output: list of KVP where keys are ms from start and vals are button value at time
    First element is start time
    '''
    # First 5 chars are ms since epoch
    start, times = string[:5], string[5:]
    # Decode 32-bit int as (22bit time, 10bit val)
    t_split = lambda s: (s // 1024, s % 1024)
    return [z85_to_bin(start)] + [t_split(s) for s in z85_decode(times)]

########################################################################################
# 16-bit format: 5char z85 = [(14bit time, 2bit val),(14bit time, 2bit val)]

def time16_encode(times):
    '''
    Encode list of (time,val) pairs as z85
    Each pair of 16bit time val elements wrapped into same 5 z85 chars
    First Element is Start Time
    '''
    # Pair up consecutive elements in list, append a (0,0) if odd length
    ziplist = lambda l: list(zip(l[::2],l[1::2]+[(0,0)]))
    # ((time,val),(time,val)) --> timevaltimeval
    timeshift = lambda ts: ts[0][0] * 2**18 + ts[0][1] * 2**16 + ts[1][0] * 4 + ts[1][1] 
    # Encode 32-bit pairs of time-val pairs as z85 string
    return z85_encode([times[0]] + [timeshift(ts) for ts in ziplist(times[1:])])

def time16_decode(string):
    '''
    Decode z85 to 16bit time-series
    Output: list of KVP where keys are decisec from start and vals are button value at time
    First element is start time
    Can have a trailing (0,0)
    '''
    # First 5 digits are the start time in ms since the epoch
    start, times = string[:5], string[5:]
    # Split a 32bit int into [(14bit time, 2bit val),(14bit time, 2bit val)]
    t16_split = lambda s: [((s//2**16)//4,(s//2**16)%4),((s%2**16)//4,(s%2**16)%4)]
    # Flatten list of lists of (time,val) and add on start, all converted to ints
    return [z85_to_bin(start)] + sum([t16_split(s) for s in z85_decode(times)],[])

def time16_to_frames(string,n_frames,frame_rate=0.04):
    '''
    Decode z85 (16bit) to frame vector
    Input: string - annotator output, n_frames: int. number of frames in whole clip
    Output: list of int - button values at each frame
    Assume 1 frame = 0.04sec
    '''
    # Convert deciseconds to frame number, assume frame = 0.04 sec
    to_frame = lambda x: [(math.ceil(t/(10*frame_rate)),v) for (t,v) in x]
    # List frames beginning with (0,0) and ending with (n_frames,1)
    # Set -> list removes any trailing (0,0) left over from decoding
    l = sorted(list(set([(0,0)]+to_frame(time16_decode(string)[1:])+[(n_frames,-1)])))
    # (frame,val) pairs exploded to list of vals up to next entry, then flattned
    return sum([[l[j][1]]*(l[j+1][0]-l[j][0]) for j in range(len(l)-1)],[])

