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
'Relaxed facial muscles, neutral eyebrow position, natural mouth expression, upright posture, gaze on learning materials, hands on notebook/keyboard.',

'Smile with teeth, eye wrinkles, raised eyebrows, forward-leaning posture, active writing, regular page-turning.',

'Furrowed brows, pressed lips, frequent blinking, flipping same pages, erasing/rewriting, erratic gaze shifts.',

'Drooping eyelids, yawning, nodding head, slowed movements, prolonged inactivity, head propped on hand.',

'Shifting expressions, sudden head turns, phone/snacks in hand, gaze away (>3s), fidgeting, reacting to distractions.'
]