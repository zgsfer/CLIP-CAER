##### 最终的提示词
class_names_5 = [
'Neutrality in learning state.',
'Enjoyment in learning state.',
'Confusion in learning state.',
'Fatigue in learning state.',
'Distraction.'
]

class_names_with_context_5 = [
'an expression of Neutrality in learning state.',
'an expression of Enjoyment in learning state.',
'an expression of Confusion in learning state.',
'an expression of Fatigue in learning state.',
'an expression of Distraction.'
]

##### 描述部分
# class_descriptor_5_face = [
# 'Focused gaze at the teacher or materials, relaxed mouth, open eyes, straight eyebrows, relaxed jaw.',

# 'Focused gaze at the teacher or materials, relaxed mouth with a gentle smile, slightly raised eyebrows.',

# 'Focused gaze at the teacher or materials, furrowed eyebrows, slightly open mouth, eyes widened or squinting.',

# 'Focused gaze at the teacher or materials, drooping eyelids, half-closed eyes, yawning, slackened mouth.',

# 'Looking away from the teacher or materials, any facial features present.'
# ]

# class_descriptor_5_body = [
# 'Writing notes and occasionally nodding.',

# 'Writing while occasionally nodding in agreement.',

# 'Writing notes but showing signs of uncertainty.',

# 'Struggling to stay awake while taking notes.',

# 'Fidgeting with objects or staring at a phone.'
# ]

# class_descriptor_5 = [
# 'Eyes lock on learning material, hand writing, steady gaze, brow smooth, head slightly nodding.',

# 'Energetic expression, wide smile, eyes brighten, eyebrows lift, eyes lock on learning material, hand writing.',

# 'Furrowed eyebrows, eyes widened or squinting, hand scratches the head, chin rests on the palm.',

# 'Mouth opens in a yawn, eyelids droop, head tilts forward, eyes lock on learning material, hand writing.',

# 'Eyes wander away, head turns slightly, fingers fidget, shoulders shift restlessly.'
# ]

##### 进行调整，生成最终的提示词
class_descriptor_5_only_face = [
'Relaxed mouth,open eyes,neutral eyebrows,smooth forehead,natural head position.',

'Upturned mouth,sparkling or slightly squinted eyes,raised eyebrows,relaxed forehead.',

'Furrowed eyebrows, slightly open mouth, squinting or narrowed eyes, tensed forehead.',

'Mouth opens in a yawn, eyelids droop, head tilts forward.',

'Averted gaze or looking away, restless or fidgety posture, shoulders shift restlessly.'
]

##### 包含上下文
class_descriptor_5 = [
'Relaxed mouth,open eyes,neutral eyebrows,no noticeable emotional changes,engaged with study materials, or natural body posture.',

'Upturned mouth corners,sparkling eyes,relaxed eyebrows,focused on course content,or occasionally nodding in agreement.',

'Furrowed eyebrows, slightly open mouth, wandering or puzzled gaze, chin rests on the palm,or eyes lock on learning material.',

'Mouth opens in a yawn, eyelids droop, head tilts forward, eyes lock on learning material, or hand writing.',

'Shifting eyes, restless or fidgety posture, relaxed but unfocused expression,frequently checking phone,or averted gaze from study materials.'
]