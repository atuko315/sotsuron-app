import sys
import base64
import inspect
import os
import io
import numpy as np
from submission_sample import SimpleAgent, dotdict, MCTS
from tqdm import tqdm
from parl.utils import logger
from connect4_game import Connect4Game
import time
from time import sleep
from datetime import datetime
import pickle
from collections import defaultdict
import collections
from random import choice
import bisect
import torch
import torch.nn as nn
import ast
from connectx_try import load_data, getStep, System, getCurrentPlayer
import collections

class DatasetManager(object):
    def __init__(self, game, path_set):
        #盤面はcanonical formで
        self.path_set = path_set
        self.game = game
        self.pattern_set = [np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 0]]
        ),
        #2group
        np.array(
            [[5, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
        ),
        np.array(
            [[5, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0]]
        ),
        #4group
        np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 0, 0]]
        ),
        np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 1, 0]]
        ),
        np.array(
            [[5, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 1, 0]]
        ),
        np.array(
            [[0, 0, 0, 5],
             [0, 1, 0, 0],
             [0, 1, 1, 0]]
        ),
        #----
        np.array(
            [[0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0]]
        ),
        np.array(
            [[0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]]
        ),
        #2group
        np.array(
            [[5, 5, 0, 0, 0],
            [5, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 5]]
        ),
        np.array(
            [[0, 0, 0, 5, 5],
            [0, 1, 0, 0, 5],
            [0, 0, 1, 0, 0],
            [5, 0, 0, 1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0, 0],
            [0,-1, 1, 1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0, 0],
            [0, 1, 1, -1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0, 0],
            [0, 1,-1, 1, 0]]
        ),
        np.array(
            [[0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0,-1, 0]]
        ),
        np.array(
            [[0, 0, 0],
            [0,-1, 0],
            [0, 1, 0],
            [0, 1, 0]]
        ),
        np.array(
            [[0, 0, 0],
            [0, 1, 0],
            [0,-1, 0],
            [0, 1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0, 0],
            [0, 1,-1, 1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0],
             [0,-1, 1, 0],
             [0,-1, 1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0,-1,-1, 0]]
        ),
        np.array(
            [[0, 0, 0, 0],
             [0, 1,-1, 0],
             [0,-1, 1, 0]]
        )
        ]
        self.board_set = []
    
    def make_board_set(self):
        #　重複のない盤面の集合をつくる
        number = len(self.path_set)
        for i in range(number):
            
            path = self.path_set[i]
            content = load_data(path)
            
            if len(content) < 5:
                _, board, _, _ = content
            else:
                _, board, _, _, _  = content
            #print(board)
            board = board.tolist()
            if board not in self.board_set:
                #print("not")
                self.board_set.append(board)
        
        #print(len(self.board_set))
    
    def retrieve_board(self, board):
        #print(board)
        datas = []
        count = 0
        for path in self.path_set:
            #print(count)
            count += 1
            h = load_data(path)
            imp, pboard, branch, fpath, importance = h
            pboard = np.array(pboard)
            #print(pboard)
            
            if (pboard == board).all():
                
                datas.append(path)
        
        return datas


    
    def remove_stone(self, board, player, column):
    #コピーで
        height, width = self.game.getBoardSize()
        available_idx = np.where(board[:, column] != 0)
        #print("ava", available_idx)
        if len(available_idx[0]) == 0 or board[available_idx[0][0]][column] != player:
            raise ValueError(
                "Can't remove column %s on board %s" % (column, self))
        #print(board[available_idx[0]][column])

        board[available_idx[0][0]][column] = 0
        return board
    
    def add_stone(self, board, player, column):
        "Create copy of board containing new stone."

        height, width = self.game.getBoardSize()
        available_idx = np.where(board[:, column] == 0)
        if len(available_idx[0]) == 0:
            raise ValueError(
                "Can't play column %s on board %s" % (column, self))

        board[available_idx[0][len(available_idx[0])-1]][column] = player
        return board

    def simple_run_length(self, board, mode='horizon'):
        '''
        列または行または斜めの行ごとに石が何個あるかを返す
        両方左下から返す

        diagonal_1:左下
        diagonal_2:右下
        '''
        board = np.array(board)
        height = len(board)
        width = len(board[0])
        win_length = self.game._base_board.win_length
        encoding = []
        
        if mode == "diagonal_1":
            maximum_number = height * width
            i = 0
            while i < maximum_number:
                seq = self.getDiagonalNumber(board, i, mode= "left_down")
              
                if seq >= 0:
                    encoding.append(seq)
                
                i = i+1 if i < width-1 else i+width
            return encoding
        
        elif mode == "diagonal_2":
            maximum_number = height * width
            i = width * (height-1)
            while i >= 0:
                seq = self.getDiagonalNumber(board, i, mode= "right_down")
                if seq >= 0:
                    encoding.append(seq)
                
                i = i - width
            for i in range(1, width):
                seq = self.getDiagonalNumber(board, i, mode= "right_down")
                if seq >= 0:
                    encoding.append(seq)
            return encoding
        
        f, s = width, height


        if mode == "vertical":
            board = np.transpose(board)
            f, s = height, width
        
       
        for i in range(f):
            extraction = np.array([board[j][i] for j in range(s)])
            count = np.count_nonzero(extraction != 0)
            encoding.append(count)
        
        if mode == "vertical":
            encoding.reverse()

        return encoding
    
    def getDiagonalNumber(self, board, n, check=False, mode="right_down"):
        height = len(board)
        width = len(board[0])
        h = int(n/width)
        w = n % width
        #右下
        if mode == "right_down":
            number = w
            record = []
            if h < w:
                number = h
            if w < h:
                sw = 0
                sh = h - w
            else:
                sw = w - h
                sh = 0
            while True:
                if sw > width -1 or sh > height -1:
                    break
                record.append(board[sh][sw])
                sw += 1
                sh += 1
            
        else:
            #左下
            if (width -1 - w< h):
                number = width -1 - w
                sw = width -1 
                sh = h - (width - 1 - w)
            else:
                number = h
                sw = w + h
                sh = 0
            
            record = []
            while True:
                if sh > height - 1 or sw < 0 :
                    break
                record.append(board[sh][sw])
                sw -= 1
                sh += 1
            
            
        record = np.array(record)
        if len(record) < self.game._base_board.win_length:
            return -1
        return np.count_nonzero(record != 0)
    
    def accumulate_variance(self):
        height, width = self.game.getBoardSize()
        number = len(self.board_set)
        horizon = 0
        vertical = 0
        diagonal_1 = 0
        diagonal_2 = 0
        for i in range(1, number):
            board = self.board_set[i]
            horizon += np.var(np.array(self.simple_run_length(board)))
            vertical += np.var(np.array(self.simple_run_length(board, mode="vertical")))
            diagonal_1 += np.var(np.array(self.simple_run_length(board, mode="diagonal_1")))
            diagonal_2 += np.var(np.array(self.simple_run_length(board, mode="diagonal_2")))
        
        return horizon.tolist(), vertical.tolist(), diagonal_1.tolist(), diagonal_2.tolist()

    
    def accumulate_run_length(self):
        height, width = self.game.getBoardSize()
        number = len(self.board_set)
        board = self.board_set[0]
        #print(board)
        horizon = np.array(self.simple_run_length(board))
        vertical = np.array(self.simple_run_length(board, mode="vertical"))
        diagonal_1 = np.array(self.simple_run_length(board, mode="diagonal_1"))
        diagonal_2 = np.array(self.simple_run_length(board, mode="diagonal_2"))
        #print(f"{horizon}, {vertical}, {diagonal_1}, {diagonal_2}")
        for i in range(1, number):
            board = self.board_set[i]
            #print(board)
            horizon += np.array(self.simple_run_length(board))
            vertical += np.array(self.simple_run_length(board, mode="vertical"))
            diagonal_1 += np.array(self.simple_run_length(board, mode="diagonal_1"))
            diagonal_2 += np.array(self.simple_run_length(board, mode="diagonal_2"))
        
        return horizon.tolist(), vertical.tolist(), diagonal_1.tolist(), diagonal_2.tolist()
    
    def grouping_connected(self, board, n, mode=4):
        height, width = self.game.getBoardSize()
        n = height * width
       
        adj = [[1, 0], [0, 1]] if mode == 4 else [[1, 0], [0, 1], [1, -1], [1, 1]]
        visited = [False for _ in range(n)]
        head = [i for i in range(n)]
        size = [1 for _ in range(n)]
        player = [[] for i in range(3)]
        data = {}
        i = 0
        while i < n:
            if visited[i]:
                continue
                
            visited[i] = True
            th = int(i / width)
            tw = i % width
            color = board[th][tw]
            if head[i] == i:
                data[i] = [i]
                player[i+1].append(i)
            if color == 0:
                continue
            for j in range(len(adj)):
                
                if th+adj[0][j] < 0 or th+adj[0][j] >= height:
                    continue
                if tw+adj[1][j] < 0 or tw+adj[1][j] >= width:
                    continue
                cn = width * (th+adj[0][j]) + (tw + adj[1][j])                
                if board[th+adj[0][j]][tw+adj[1][j]] == color:
                    head[cn] = head[i]
                    size[head[i]] += 1
                    data[head[i]].append(cn)
        
        distribution_1 = [size[k] for k in player[0]]
        distribution_1 = collections.Counter(distribution_1)
        distribution_2 = [size[k] for k in player[1]]
        distribution_2 = collections.Counter(distribution_2)
        
        return size, player, data
    
    def match_pattern(self, board, pattern):
        #その形が含まれているか
        #print(f"{pattern}\n=====================\n{board}")
        board = np.array(board)
        contain_indices = []
        pure_indices = []
        number = np.count_nonzero(pattern!=0)
        ph, pw = len(pattern), len(pattern[0])
        height = len(board)
        width = len(board[0])
        eboard = np.full((height+1, width+2), -2)
        eboard[1:, 1: len(eboard)+1] = board
        #print(eboard)
        purepattern = pattern.copy()
        purepattern[abs(purepattern) != 1] = 0
        purepattern[abs(purepattern) == 1] = 1
        #print(purepattern)
        #どっちや？？
        for i in range(height + 1 - ph + 1):
            for j in range(width + 2 - pw + 1):
                submatrix = eboard[i:i+ph, j:j+pw]
                inverse = submatrix * -1
                #print(submatrix)
                #print(submatrix * purepattern)
                if (submatrix * purepattern == pattern).all() or (inverse * purepattern == pattern).all():
                    #含まれてる
                    #if (i!=0 and board[i-1][j]==0):
                    #    contain_indices.append((i)*width+j)
                    contain_indices.append((i+1)*width+j)
                    #print("contain")
                    #print(submatrix * purepattern)
                    #print(inverse * purepattern)
                    #print("------------------")
                    margin = submatrix - pattern
                    inverse_margin = inverse - pattern
                    surrounding = np.sum((abs(margin)!=0) | (abs(margin) <= 3))
                    inverse_surrounding = np.sum((abs(inverse_margin)!=0) | (abs(inverse_margin) <= 3))
                    if (surrounding == 0 or inverse_surrounding==0) :
                        
                        pure_indices.append(i*width+j)
                        #print("pure")
        
        return contain_indices, pure_indices
    
    def accumulate_pattern(self, pattern):
        #登場局面数 
        contain_count = 0
        pure_count = 0
        height, width = self.game.getBoardSize()
        number = len(self.board_set)
       
        for i in range(number):
            
            board = self.board_set[i]
            
            #print(board)
            contain_indices, pure_indices = self.match_pattern(board, pattern)
            contain_count += 1 if len(contain_indices) else 0
            pure_count += 1 if len(pure_indices) else 0
 
        return contain_count, pure_count
    
    def multiple_accumulate_pattern(self):
        print("pattern")
        for i in range(len(self.pattern_set)):
            contain_count, pure_count = self.accumulate_pattern(self.pattern_set[i])
            print(contain_count, pure_count)
    
    def generate_alternative_board(self, board, change=1, verbose=False):
        height, width = self.game.getBoardSize()
        print(board)
        step = getStep(board)
        assert change <= step
        count = change
        player = -getCurrentPlayer(board) #別に逆のプレイヤーからでもOK
        new_board = board.copy()
        while count > 0:
            print(f"step: {change-count+1}")
            ravailable_idx = []
            for c in range (width):
                tmp = np.where(new_board[:, c] != 0)
                if tmp[0].size > 0 and new_board[tmp[0][0]][c] == player:
                    ravailable_idx.append(c)
            if len(ravailable_idx) == 0:
                player = -player
                for c in range (width):
                    tmp = np.where(new_board[:, c] != 0)
                    if tmp[0].size > 0 and new_board[tmp[0][0]][c] == player:
                        ravailable_idx.append(c)
            if verbose:
                print(f"available:{ravailable_idx}")
            
            #removeする列rを決める rは前のat
            r = choice(ravailable_idx)
            if verbose:
                print("choose column to remove")
                r = int(input())
                while r not in ravailable_idx:
                    print("unavailable")
                    r = int(input())
                print(f"choice: {r}")
            #remove 
            new_board = self.remove_stone(new_board, player, r)
          
            # r 以外で石を置く列aを決める
            aavailable_idx = []
            for c in range (width):
                tmp = np.where(new_board[:, c] != 0)
                if 0 not in tmp[0]:
                    aavailable_idx.append(c)
           
            assert len(aavailable_idx) > 0
            if len(aavailable_idx) == 0:
                break
            if verbose:
                print(f"available:{aavailable_idx}")
            if r in aavailable_idx:
                aavailable_idx.remove(r)
            a = choice(aavailable_idx)
            if verbose:
                print("choose column to add")
                a = int(input())
                while a not in aavailable_idx:
                    print("unavailable")
                    a = int(input())
            new_board = self.add_stone(new_board, player, a)
            #print(new_board)
            # add
            player = - player
            count -= 1
        print(new_board)
        return  new_board.copy()
    
    def label_board(self, board, path, step, system, threshold=0.6):
        simp = system.getImportance(board, 1, path, step)
        wimp = system.getImportance(board, -1, path, step)
        if simp > threshold:
            if wimp > threshold:
                label = 1
            else:
                label = 2
        else:
            if wimp > threshold:
                label = 3
            else:
                label = 4
        
        return label





    
    def register_alternative_board(self, path, system, dirname, change=1, threshold=0.6, verbose=False):
        '''
        既存のデータセットの盤面を変更した盤面を作る
        systemによるimportanceの判断も再度行う
        データはalterにラベル付きで格納
        .alterは（元の盤面、生成された盤面、元のパス、システムとかのパス)
        
        '''
        
        height, width = self.game.getBoardSize()
        
        _, board, _, fpath, _ = load_data(path)
        #print(board)
        '''
        step = getStep(board)
        assert change <= step
        count = change
        crec = -1 #　直前に石を置き直した列を動かさない
        player = -getCurrentPlayer(board) #別に逆のプレイヤーからでもOK
        new_board = board.copy()
        
        while count:
            print(f"step: {change-count+1}")
            ravailable_idx = []
            for c in range (width):
                tmp = np.where(new_board[:, c] != 0)
                if tmp[0].size > 0 and new_board[tmp[0][0]][c] == player:
                    ravailable_idx.append(c)
            if len(ravailable_idx) == 0:
                player = -player
                for c in range (width):
                    tmp = np.where(new_board[:, c] != 0)
                    if tmp[0].size > 0 and new_board[tmp[0][0]][c] == player:
                        ravailable_idx.append(c)
            if verbose:
                print(f"available:{ravailable_idx}")
            
            #removeする列rを決める rは前のat
            r = choice(ravailable_idx)
            if verbose:
                print("choose column to remove")
                r = int(input())
                while r not in ravailable_idx:
                    print("unavailable")
                    r = int(input())
                print(f"choice: {r}")
            #remove 
            new_board = self.remove_stone(new_board, player, r)
          
            # r 以外で石を置く列aを決める
            aavailable_idx = []
            for c in range (width):
                tmp = np.where(new_board[:, c] != 0)
                if 0 not in tmp[0]:
                    aavailable_idx.append(c)
           
            assert len(aavailable_idx) > 0
            if len(aavailable_idx) == 0:
                break
            if verbose:
                print(f"available:{aavailable_idx}")
            if r in aavailable_idx:
                aavailable_idx.remove(r)
            a = choice(aavailable_idx)
            if verbose:
                print("choose column to add")
                a = int(input())
                while a not in aavailable_idx:
                    print("unavailable")
                    a = int(input())
            new_board = self.add_stone(new_board, player, a)
            #print(new_board)
            # add
            player = - player
            count -= 1
        '''
        step = getStep(board)
        new_board = self.generate_alternative_board(board, change=change, verbose=verbose)

           
        data = (board, new_board, path, fpath)
        simp = system.getImportance(new_board, 1, fpath, step)
        
        s = dirname + '/important' if simp > threshold else dirname + '/trivial'
        wimp = system.getImportance(new_board, -1, fpath, step)
        s = s + '/important' if wimp > threshold else s + '/trivial'
        if verbose:
            print(f"simp: {simp}, wimp: {wimp}")


        new_path = './' + s
        now = datetime.now()
        new_path += '/{:04}{:02}{:02}{:02}{:02}{:02}.alter'.format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)
        with open(new_path, mode='wb') as f:
            pickle.dump(data, f)
        
        return new_board, simp, wimp
    
    def make_pattern_set(self, pattern, mode="contain"):
        data = defaultdict(lambda: [])
        for board in self.board_set:
            contain_indices, pure_indices = self.match_pattern(board, pattern)
            indices = contain_indices if mode == "contain" else pure_indices
            for i in indices:
                data[i].append(board)
        
        return data
    
    def make_pattern_path_set(self, pattern, mode="contain"):
        data = []
        for path in self.path_set:
            content = load_data(path)
            if len(content) < 5:
                importance, board, brance, fpath = content
            else:
                imp, board, branch, fpath, importance = content
            
            contain_indices, pure_indices = self.match_pattern(board, pattern)
            indices = contain_indices if mode == "contain" else pure_indices
            if len(indices) > 0:
                data.append(path)
        
        return data
    
    def detect_relative_distance(self, pa, ca):
        '''
        左から-1, 0, 1
        '''
        l = 0
        if ca < pa:
            l = -1
        elif ca > pa:
            l = 1
        
        return (l, abs(pa-ca))
    
    def detect_relative_traj(self, board,  key_c, path, system, analist, step=5):

        traj = []
        height, width = self.game.getBoardSize()
        content = load_data(path)
        if len(content) < 5:
            importance, board, brance, fpath = content
        else:
            imp, board, branch, fpath, importance = content
        
        bstep = getStep(board)
        zflag = True if analist == 0 else False
        fcontent = load_data(fpath)
        system.turn = fcontent[len(fcontent)-1][0]
        tmp = fcontent[bstep] # 注目する部分
        tboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        curPlayer = getCurrentPlayer(board)
        
        
        vboard = board.copy()
        vcanonicalBoard = self.game.getCanonicalForm(vboard, curPlayer)
        vs = self.game.stringRepresentation(vcanonicalBoard)
        
        vstep = bstep # countは差分で得られる
        print(board)
        print(bstep)
        if zflag:
            analist = getCurrentPlayer(board)
        counts = system.getPastCount(fpath, bstep, vboard, analist)
        #print(self.getPastValueNoModification( path, step, vboard, 1))
        if self.game.getGameEnded(board, curPlayer):
           return traj
            
        
        if analist == 1:
            if vs not in sVs.keys():
                return traj
        else:
            if vs not in bVs.keys():
                return traj
        
        
        vplayer = curPlayer
    
        for i in range(step):
            if zflag:
                analist = vplayer

            #print(vboard)
            #print(vvalue)
            #print("--------")
        
            valids = self.game.getValidMoves(vboard, vplayer)
            counts = system.getPastCount(fpath, bstep, vboard, analist)
            action = np.argmax(system.getPastActionProb(fpath, bstep, vboard, 
                                                    analist, counts = counts))
            print(action)
            if valids[action] == 0 or sum(counts) == 0:
                return traj
            #手を抽象化する
            
            traj.append(self.detect_relative_distance(key_c, action))
            vnextBoard, vplayer = self.game.getNextState(vboard, vplayer, action)
            vcanonicalBoard = self.game.getCanonicalForm(vboard, -vplayer)
            vs = self.game.stringRepresentation(vcanonicalBoard)
        
            vstep += 1
            if analist == 1:
                if vs not in sVs.keys():
                    return traj
            else:
                if vs not in bVs.keys():
                    return traj
        
            if self.game.getGameEnded(vnextBoard, vplayer):
                # end
                return traj
            
            vboard = vnextBoard
            print(vboard)

        print(traj)   
        
        return traj
    


    
    def collect_pattern_vector(self, pattern, system, analist, step=5, mode="contain"):
        '''
        左から1, 2, 3
        基本的にpatternがあるやつだけにして
        '''
        size = 0
        height, width = self.game.getBoardSize()
        traj_set = []
        for path in self.path_set:
            tmp_traj = self.detect_pattern_vector(pattern, path, system, analist, step=step, mode=mode)
            traj_set.append(tmp_traj)
        return traj_set




    
    def detect_pattern_vector(self, pattern, path, system, analist, step=5, mode="contain"):
        '''
        traj
        パターン複数の場合も気にせずにやっとるので
        '''
        traj = []
        height, width = self.game.getBoardSize()
        content = load_data(path)
        if len(content) < 5:
            importance, board, brance, fpath = content
        else:
            imp, board, branch, fpath, importance = content
        
        contain_indices, pure_indices = self.match_pattern(board, pattern)

        for c in contain_indices:
            #c複数の場合
            key_w = int(c/width)
            key_c = c % width
            tmp_traj = self.detect_relative_traj(board, key_c, path, system, analist=analist, step=step)
            traj.append(tmp_traj)

        
        
        return traj
    


    
    def collect_pattern_vector(self, pattern, system, analist, step=5, mode="contain"):
        '''
        左から1, 2, 3
        基本的にpatternがあるやつだけにして
        '''
        size = 0
        height, width = self.game.getBoardSize()
        traj_set = []
        for path in self.path_set:
            tmp_traj = self.detect_pattern_vector(pattern, path, system, analist, step=step, mode=mode)
            traj_set.append(tmp_traj)
        return traj_set

                

            
    
    def detect_actual_reach(self, path, system):
        content = load_data(path)
        
        if len(content) < 5:
            importance, vboard, brance, fpath = content
        else:
            imp, vboard, branch, fpath, importance = content
        
        h = load_data(fpath)
        if len(h) - 2 < 0:
            None
        tmp = h[len(h)-2]
        fboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        
        valid = self.game.getValidMoves(fboard, getCurrentPlayer(fboard))
        valid = [i  for i in range(len(valid)) if valid[i]]
        reach = []
        for a in valid:
            vboard = self.add_stone(fboard.copy(), getCurrentPlayer(fboard), a)
            vf = system.detectFatalStone(vboard, per_group=True)
            if vf:
                reach.extend(vf)
        
        return reach
    
    def check_convergence(self, boards, reach, path, bstep, system, analist, mode="analysis"):
        '''
        bcount, bfcount, bfdcount   の順 確率で出力
        あとでgroupを保持する版もつくる
        stepはbstep

        focus モードは　グループと石を集めて上位いくつかを 石は上位４つ、グループは上位3つ
        '''
        
        if mode== "show" or mode == "focus":
            gd = defaultdict(lambda: 0)
            gs = defaultdict(lambda: 0)
        size = len(boards)

        if size < 1:
            if mode == "show" or mode == "focus":
                return [], []
            return 0, 0, 0
        #print(size)
        c=0
        bcount = 0
        bfcount = 0
        bfdcount = 0

        for b in boards:
            #print(f"{c}/{size}")
            if mode == "show" or mode == "focus":
                answer = self.check_convergence_per_board(b, reach, path, bstep, system, analist, mode=mode)
                group, stones = answer
                if group:
                    for g in group:
                        gd[str(g)] += 1

                if stones:
                    for s in stones:
                        gs[s] += 1
                
                continue


                
            c+=1
            bc, bf, bfd  = self.check_convergence_per_board(b, reach, path, bstep, system, analist)
            bcount += bc
            bfcount += bf
            bfdcount += bfd
        
        if mode == "show":
            
            height, width = self.game.getBoardSize()
            visual = [0 if i not in collections.Counter(gs).keys() else collections.Counter(gs)[i]
                        for i in range(height * width)]
            visual = np.array(visual).reshape(height, width)
            print(visual)
            danger = max(gd, key=gd.get)
            print(danger)

            return gd, gs
        elif mode == "focus":
            height, width = self.game.getBoardSize()
            visual = [0 if i not in collections.Counter(gs).keys() else collections.Counter(gs)[i]
                        for i in range(height * width)]
            visual = np.array(visual).reshape(height, width)
            print(visual)
            
            gs_sorted = sorted(dict(gs).items(), reverse=True, key=lambda x : x[1])
            
            if len(gs_sorted) < 4:
                gs4 = [gs_sorted[i][0] for i in range(len(gs_sorted))]
            else:
                gs4 = [gs_sorted[i][0] for i in range(4)]
            
            gd_sorted = sorted(dict(gd).items(), reverse=True, key=lambda x : x[1])
            
            if len(gd_sorted) < 2:
                gd2 = [gd_sorted[i][0] for i in range(len(gd_sorted))]
            else:
                gd2 = [gd_sorted[i][0] for i in range(2)]
            gd2 = [eval(g) for g in gd2]
            print(gd2)
            fu = np.unique(gs4.copy()).tolist() if gs4 else [-1]
            ru = np.unique(reach.copy()).tolist() if reach else [-2]
            bfdcount = 0
            bfcount = 0
            if len(set(ru)) > 0:
                bfdcount = (len(set(fu) & set(ru))) / 4
            #print(fu, ru, len(set(fu) & set(ru)))つくる
            if gd2:
                for g in gd2:
                    for i in range(len(reach)):
                        r = reach[i]
                        if set(r).issubset(set(g)):
                            bfcount = 1
            
            return bfcount, bfdcount






        
        rate = bcount / size
        frate = bfcount / bcount if bcount > 0 else 0
        fdrate = bfdcount / bcount if bcount > 0 else 0

        return rate, frate, fdrate

    
    def check_convergence_per_board(self, board, reach, path, bstep, system, analist, mode="analysis"):
        '''
        bcount, bfcount, bfdcount   の順
        あとでgroupを保持する版もつくる
        show modeだとgroup と石を返します
        focus modeだと fatalを返す
        '''
        bcount = 0
        bfcount = 0
        bfdcount = 0
        #print(board)
        hot = system.detectHotState(board, analist, path, bstep, toend=True)
        #print(hot[1])
        if hot[1] == None:
            if mode == "show":
                
                return None, []
            elif mode == "focus":
                return None, None
            return bcount, bfcount, bfdcount
        
        end = self.game.getGameEnded(hot[0], getCurrentPlayer(hot[0]))
        if end:
            #print("reach")
            #print(reach)
            
            bcount = 1
            fatal = system.detectFatalStone(hot[0], per_group=True)
            if mode == "show" or mode == "focus":
                fu = np.unique(fatal.copy()).tolist() if fatal else []
                return fatal, fu
            #print("fatal")
            #print(fatal)
            fu = np.unique(fatal.copy()).tolist() if fatal else [-1]
            ru = np.unique(reach.copy()).tolist() if reach else [-2]
            if len(set(ru)) > 0:
                bfdcount = (len(set(fu) & set(ru))) / 4
            #print(fu, ru, len(set(fu) & set(ru)))つくる
            if fatal:
                for g in fatal:
                    for i in range(len(reach)):
                        r = reach[i]
                        if set(r).issubset(set(g)):
                            bfcount = 1
                '''
                gs = str(g)
                #print(fatal_group.keys())
                if gs not in fatal_group.keys():
                    fatal_group[gs] = 1
                else:
                    fatal_group[gs] += 1
                '''
        if mode == "show":
            return [], []
        elif mode == "focus":
            return bfcount, bfdcount
        return bcount, bfcount, bfdcount      
    
    
                        
        
        

    
    def collect_promising_per_step(self, board, path, system, analist, baseline=3, fix=-1):
        content = load_data(path)
        if len(content) < 5:
            importance, fboard, brance, fpath = content
        else:
            imp, fboard, branch, fpath, importance = content
        
        max_step = len(load_data(fpath))  - 2
        if analist == 0:
            analist = getCurrentPlayer(board)
        valid = self.game.getValidMoves(board, getCurrentPlayer(board))
        valid = [i  for i in range(len(valid)) if valid[i]]
        l = len(valid) if len(valid) < baseline else baseline
        #print(fpath, getStep(board))
        bstep = getStep(board)
        bstep = max_step if bstep > max_step else bstep
        if fix != -1:
            bstep = fix
        
        counts = system.getPastCount(fpath, bstep, board, analist)
        counts = np.argsort(np.array(counts))
        counts = [c for c in counts if c in valid]
        counts = counts[-l:]
        fboards = []

        for c in counts:
                fboards.append(system.add_stone(board.copy(), getCurrentPlayer(board), c))
    
        return fboards
    
    def collect_promising(self, board, path, system, analist, step, baseline=3, fix=-1):
        #print("start")
        #print(board)
        assert step > 0
        boards = []
        boards = self.collect_promising_per_step(board.copy(), path, system, analist, baseline=baseline, fix=fix)
        if step == 1:
            return boards
        result = []
        for b in boards:
            tmp = self.collect_promising(b.copy(), path, system, analist, step-1, baseline=baseline, fix=fix)
            
            if len(tmp) > 0:
                result.extend(tmp)

        #print("success")
        return result
        
            
    
    def hot_states_one_way(self, board, path, system, analist, step, baseline=3, mode="analysis", fix=-1):
        '''
        step分先のを集めてそこからはhotstatesつまり、step分先の盤面数
        path step の stepはbstep
        collect: 絞り込みありで先読み
        check_convergence: 絞り込みなし
        '''
        assert step > 0
        reach = self.detect_actual_reach(path, system)
        #print(reach)
        bstep = getStep(board)
        if fix != -1:
            bstep = fix
    
        content = load_data(path)
        if len(content) < 5:
            importance, vboard, brance, fpath = content
        else:
            imp, vboard, branch, fpath, importance = content
        
        boards = self.collect_promising(board, path, system, analist, step, baseline=baseline, fix = bstep)
        
        if mode == "show":
            gd, gs = self.check_convergence(boards, reach, fpath, bstep, system, analist, mode="show")
            return gd, gs
        elif mode == "focus":
            bfcount, bfdcount = self.check_convergence(boards, reach, fpath, bstep, system, analist, mode="focus")
            return bfcount, bfdcount

        rate, frate, fdrate = self.check_convergence(boards, reach, fpath, bstep, system, analist)


        return rate, frate, fdrate

    def hot_states_two_ways(self, board, path, system, analist, step, baseline=1, promising=3, mode="compare"):
        '''
        一手先読みのbaseline変えたいときはpromising
        modeをrandomにするとbaselineはランダムになる
        今いじってるからきをつけろ
        '''
        print(board)

        content = load_data(path)
        if len(content) < 5:
            importance, vboard, brance, fpath = content
        else:
            imp, vboard, branch, fpath, importance = content

        if mode == "ramdom":
            baseline = choice([i for i in range(len(board[0]))])
        
        bstep = getStep(board)
        valid = system.game.getValidMoves(board, getCurrentPlayer(board))
        valid = [i  for i in range(len(valid)) if valid[i]]
        if len(valid) < baseline+1:
            # これもとりあえず最下位をわたしとく
            baseline = len(valid) - 1
        
        reach = self.detect_actual_reach(path, system)
        print(reach)
        
        best = system.getImportantAction(board, analist, fpath, getStep(board), 0)
        second = system.getImportantAction(board, analist, fpath, getStep(board), baseline)
        #print(best, second)

        best_board = self.add_stone(board.copy(), getCurrentPlayer(board), best)
        second_board = self.add_stone(board.copy(), getCurrentPlayer(board), second)

        if mode == "focus":
            bfcount, bfdcount = self.hot_states_one_way(best_board, path, system, analist, step=step, baseline=promising, fix=bstep, mode="focus")
            sfcount, sfdcount = self.hot_states_one_way(second_board, path, system, analist, step=step, baseline=promising, fix=bstep, mode="focus")
            return bfcount, bfdcount, sfcount, sfdcount


        brate, bfrate, bfdrate = self.hot_states_one_way(best_board, path, system, analist, step=step, baseline=promising, fix=bstep)
        srate, sfrate, sfdrate = self.hot_states_one_way(second_board, path, system, analist, step=step, baseline=promising,fix=bstep)
        #print(brate, bfrate, bfdrate, srate, sfrate, sfdrate)
        print("*****************************************")
        return brate, bfrate, bfdrate, srate, sfrate, sfdrate
    
    def collect_two_ways(self, system, analist, step=3, baseline=1, promising=3,mode="compare"):
        #size = len(self.path_set)
        #print(f"size: {size}")
        size = 0
        ave_brate = 0
        ave_bfrate = 0
        ave_bfdrate = 0

        ave_srate = 0
        ave_sfrate = 0
        ave_sfdrate = 0
        
        for p in self.path_set:
            content = load_data(p)
            if len(content) < 5:
                importance, board, brance, fpath = content
            else:
                imp, board, branch, fpath, importance = content

            
            #if getCurrentPlayer(board) != analist:
            #    continue
            if getStep(board) < 15 or getStep(board) > 20:
                    continue
            size += 1
            
            if mode == "focus":
                bfcount, bfdcount, sfcount, sfdcount = self.hot_states_two_ways(board, p, system, analist, step=step, baseline=baseline, promising=promising, mode="focus")
                ave_bfrate += bfcount
                ave_bfdrate += bfdcount
                ave_sfrate += sfcount
                ave_sfdrate += sfdcount
                continue
            brate, bfrate, bfdrate, srate, sfrate, sfdrate = self.hot_states_two_ways(board, p, system, analist, step=step, baseline=baseline, promising=promising)
            

            ave_brate += brate
            ave_bfrate += bfrate
            ave_bfdrate += bfdrate
            ave_srate += srate
            ave_sfrate += sfrate
            ave_sfdrate += sfdrate
        
        if size == 0:
            if mode == "focus":
                return 0, 0, 0, 0, 0
            return 0, 0, 0, 0, 0, 0, 0
        
        ave_brate /= size
        ave_bfrate /= size
        ave_bfdrate /= size
        ave_srate /= size
        ave_sfrate /= size
        ave_sfdrate /= size
        if mode == "focus":
            return ave_bfrate, ave_bfdrate, ave_sfrate, ave_sfdrate, size
        return ave_brate, ave_bfrate, ave_bfdrate, ave_srate, ave_sfrate, ave_sfdrate, size

    def collect_one_way(self, system, analist, step=3, promising=3, mode="analysis"):
            '''
            こっちはfocusあり
            '''
            #size = len(self.path_set)
            #print(f"size: {size}")
            size = 0
            ave_brate = 0
            ave_bfrate = 0
            ave_bfdrate = 0

            
            for p in self.path_set:
                content = load_data(p)
                if len(content) < 5:
                    importance, board, brance, fpath = content
                else:
                    imp, board, branch, fpath, importance = content

                if getStep(board) < 15 or getStep(board) > 20:
                    continue
                #if abs(analist) == 1:
                #    if getCurrentPlayer(board) != analist:
                #        continue

                size += 1
                if mode == "focus":
                    bfcount, bfdcount = self.hot_states_one_way(board, p, system, analist, step=step, baseline=promising, mode="focus")
                    ave_bfrate += bfcount
                    ave_bfdrate += bfdcount
                    continue

                
                brate, bfrate, bfdrate = self.hot_states_one_way(board, p, system, analist, step=step, baseline=promising)
                ave_brate += brate
                ave_bfrate += bfrate
                ave_bfdrate += bfdrate
                
            
            if size == 0:
                if mode == "focus":
                    
                    return 0, 0
                return 0, 0, 0
            
            ave_brate /= size
            ave_bfrate /= size
            ave_bfdrate /= size
            if mode == "focus":
                print(f"size: {size}")
                return ave_bfrate, ave_bfdrate, size
            
            return ave_brate, ave_bfrate, ave_bfdrate



