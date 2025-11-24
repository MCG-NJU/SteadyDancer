import random
import copy
import numpy as np

def pose_aug_diff(pose, size, offset, scale, aspect_ratio_range, add_aug=True):

    h, w = size
    if h >= w: 
        new_h = int(h*1024/w)
        new_w = 1024
    else:
        new_h = 1024
        new_w = int(w*1024/h)

    # bodies = pose[0]['bodies']
    # hands = pose[0]['hands']
    # candidate = bodies['candidate']

    # center = candidate[0]
    # pose_refer = copy.deepcopy(pose[0])
    
    if add_aug:

        # offset = random.uniform(*offset)
        offset_x, offset_y = offset
        offset_x = random.uniform(*offset_x)
        offset_y = random.uniform(*offset_y)
        scale = random.uniform(*scale)
        asp_ratio = random.uniform(*aspect_ratio_range)

        # for p in pose:

        #     # adjust ratio
        #     p['bodies']['candidate'][:, 0] = p['bodies']['candidate'][:, 0] * asp_ratio
        #     p['hands'][:, :, 0] = p['hands'][:, :, 0] * asp_ratio

        #     # scale the pose
        #     p['hands'] *= scale
        #     p['bodies']['candidate'] *= scale

        #     # move the center of pose
        #     p['hands'] += offset
        #     p['bodies']['candidate'] += offset

        # run align
        # pose_aug = run_align_video_with_filterPose_translate_smooth_woload(pose, pose_refer, size, frame_num=len(pose), align_pose=True)

        _pose = copy.deepcopy(pose)
        
        # adjust ratio
        pose['bodies']['candidate'][:, 0] = pose['bodies']['candidate'][:, 0] * asp_ratio
        pose['hands'][:, :, 0] = pose['hands'][:, :, 0] * asp_ratio

        # scale the pose
        pose['hands'] *= scale
        pose['bodies']['candidate'] *= scale

        # # move the center of pose
        # # offset_x, offset_y = offset
        # # pose['hands'] += offset
        # pose['hands'][:, :, 0] += offset_x
        # pose['hands'][:, :, 1] += offset_y
        # # pose['bodies']['candidate'] += offset
        # pose['bodies']['candidate'][:, 0] += offset_x
        # pose['bodies']['candidate'][:, 1] += offset_y

        _offset = _pose['bodies']['candidate'][1] - pose['bodies']['candidate'][1]

        pose['bodies']['candidate'] += _offset[np.newaxis, :]
        pose['faces'] += _offset[np.newaxis, np.newaxis, :]
        pose['hands'] += _offset[np.newaxis, np.newaxis, :]

        return pose


def pose_aug_same(pose, size, offset, scale, asp_ratio, add_aug=True):

    h, w = size
    if h >= w: 
        new_h = int(h*1024/w)
        new_w = 1024
    else:
        new_h = 1024
        new_w = int(w*1024/h)

    # bodies = pose[0]['bodies']
    # hands = pose[0]['hands']
    # candidate = bodies['candidate']

    # center = candidate[0]
    # pose_refer = copy.deepcopy(pose[0])
    
    if add_aug:

        # offset = random.uniform(*offset)
        # scale = random.uniform(*scale)
        # asp_ratio = random.uniform(*aspect_ratio_range)

        # for p in pose:

        #     # adjust ratio
        #     p['bodies']['candidate'][:, 0] = p['bodies']['candidate'][:, 0] * asp_ratio
        #     p['hands'][:, :, 0] = p['hands'][:, :, 0] * asp_ratio

        #     # scale the pose
        #     p['hands'] *= scale
        #     p['bodies']['candidate'] *= scale

        #     # move the center of pose
        #     p['hands'] += offset
        #     p['bodies']['candidate'] += offset

        # run align
        # pose_aug = run_align_video_with_filterPose_translate_smooth_woload(pose, pose_refer, size, frame_num=len(pose), align_pose=True)

        _pose = copy.deepcopy(pose)
        
        # adjust ratio
        pose['bodies']['candidate'][:, 0] = pose['bodies']['candidate'][:, 0] * asp_ratio
        pose['hands'][:, :, 0] = pose['hands'][:, :, 0] * asp_ratio

        # scale the pose
        pose['hands'] *= scale
        pose['bodies']['candidate'] *= scale

        # # move the center of pose
        # offset_x, offset_y = offset
        # # pose['hands'] += offset
        # pose['hands'][:, :, 0] += offset_x
        # pose['hands'][:, :, 1] += offset_y
        # # pose['bodies']['candidate'] += offset
        # pose['bodies']['candidate'][:, 0] += offset_x
        # pose['bodies']['candidate'][:, 1] += offset_y

        _offset = _pose['bodies']['candidate'][1] - pose['bodies']['candidate'][1]

        pose['bodies']['candidate'] += _offset[np.newaxis, :]
        pose['faces'] += _offset[np.newaxis, np.newaxis, :]
        pose['hands'] += _offset[np.newaxis, np.newaxis, :]

        return pose