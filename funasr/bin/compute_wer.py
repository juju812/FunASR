import os
import numpy as np
import sys

class Wer(object):
    def __init__(self, ref_file, cer_detail_file="./wer"):
        self.ref_dict = {}
        with open(ref_file, 'r') as ref_reader:
            for line in ref_reader:
                key = line.strip().split()[0]
                value = line.strip().split()[1:]
                self.ref_dict[key] = value
        
        self.rst = {
            'Wrd': 0,
            'Corr': 0,
            'Ins': 0,
            'Del': 0,
            'Sub': 0,
            'Snt': 0,
            'Err': 1.0,
            'S.Err': 1.0,
            'wrong_words': 0,
            'wrong_sentences': 0
        }
        self.cer_detail_writer = open(cer_detail_file, 'w')
        self.hyp_token_num = 0


    def compute_wer(self, key, token):
        self.hyp_token_num += 1
        if key in self.ref_dict:
            out_item = self.compute_wer_by_line(token, self.ref_dict[key])
            self.rst['Wrd'] += out_item['nwords']
            self.rst['Corr'] += out_item['cor']
            self.rst['wrong_words'] += out_item['wrong']
            self.rst['Ins'] += out_item['ins']
            self.rst['Del'] += out_item['del']
            self.rst['Sub'] += out_item['sub']
            self.rst['Snt'] += 1
            if out_item['wrong'] > 0:
                self.rst['wrong_sentences'] += 1
            self.cer_detail_writer.write(key + self.print_cer_detail(out_item) + '\n')
            self.cer_detail_writer.write("ref:" + '\t' + "".join(self.ref_dict[key]) + '\n')
            self.cer_detail_writer.write("hyp:" + '\t' + "".join(token) + '\n')

    def summary(self):
        if self.rst['Wrd'] > 0:
            self.rst['Err'] = round(self.rst['wrong_words'] * 100 / self.rst['Wrd'], 2)
        if self.rst['Snt'] > 0:
            self.rst['S.Err'] = round(self.rst['wrong_sentences'] * 100 / self.rst['Snt'], 2)

        self.cer_detail_writer.write('\n')
        self.cer_detail_writer.write("%WER " + str(self.rst['Err']) + " [ " + str(self.rst['wrong_words'])+ " / " + str(self.rst['Wrd']) +
                                ", " + str(self.rst['Ins']) + " ins, " + str(self.rst['Del']) + " del, " + str(self.rst['Sub']) + " sub ]" + '\n')
        self.cer_detail_writer.write("%SER " + str(self.rst['S.Err']) + " [ " + str(self.rst['wrong_sentences']) + " / " + str(self.rst['Snt']) + " ]" + '\n')
        self.cer_detail_writer.write("Scored " + str(self.hyp_token_num) + " sentences, " + str(self.hyp_token_num - self.rst['Snt']) + " not present in hyp." + '\n')
        if self.rst['Wrd'] > 0:
            return max(0, 1 - self.rst['wrong_words'] / self.rst['Wrd'])
        else:
            return 0

    @staticmethod  
    def compute_wer_by_line(hyp, ref):
        hyp = list(map(lambda x: x.lower(), hyp))
        ref = list(map(lambda x: x.lower(), ref))

        len_hyp = len(hyp)
        len_ref = len(ref)

        cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

        ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

        for i in range(len_hyp + 1):
            cost_matrix[i][0] = i
        for j in range(len_ref + 1):
            cost_matrix[0][j] = j

        for i in range(1, len_hyp + 1):
            for j in range(1, len_ref + 1):
                if hyp[i - 1] == ref[j - 1]:
                    cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
                else:
                    substitution = cost_matrix[i - 1][j - 1] + 1
                    insertion = cost_matrix[i - 1][j] + 1
                    deletion = cost_matrix[i][j - 1] + 1

                    compare_val = [substitution, insertion, deletion]

                    min_val = min(compare_val)
                    operation_idx = compare_val.index(min_val) + 1
                    cost_matrix[i][j] = min_val
                    ops_matrix[i][j] = operation_idx

        match_idx = []
        i = len_hyp
        j = len_ref
        rst = {
            'nwords': len_ref,
            'cor': 0,
            'wrong': 0,
            'ins': 0,
            'del': 0,
            'sub': 0
        }
        while i >= 0 or j >= 0:
            i_idx = max(0, i)
            j_idx = max(0, j)

            if ops_matrix[i_idx][j_idx] == 0:  # correct
                if i - 1 >= 0 and j - 1 >= 0:
                    match_idx.append((j - 1, i - 1))
                    rst['cor'] += 1

                i -= 1
                j -= 1

            elif ops_matrix[i_idx][j_idx] == 2:  # insert
                i -= 1
                rst['ins'] += 1

            elif ops_matrix[i_idx][j_idx] == 3:  # delete
                j -= 1
                rst['del'] += 1

            elif ops_matrix[i_idx][j_idx] == 1:  # substitute
                i -= 1
                j -= 1
                rst['sub'] += 1

            if i < 0 and j >= 0:
                rst['del'] += 1
            elif j < 0 and i >= 0:
                rst['ins'] += 1

        match_idx.reverse()
        wrong_cnt = cost_matrix[len_hyp][len_ref]
        rst['wrong'] = wrong_cnt

        return rst

    def print_cer_detail(rst):
        return ("(" + "nwords=" + str(rst['nwords']) + ",cor=" + str(rst['cor'])
                + ",ins=" + str(rst['ins']) + ",del=" + str(rst['del']) + ",sub="
                + str(rst['sub']) + ") corr:" + '{:.2%}'.format(rst['cor']/rst['nwords'])
                + ",cer:" + '{:.2%}'.format(rst['wrong']/rst['nwords']))

