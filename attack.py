from benchmark_1N import EnrollTemplates, SearchTemplatePrivacyAttack   
import numpy as np

def prepare_attack_data(member_data, nonmember_data, model, transform=None,
                         embedding_agg=np.mean, random_state=42, n_samples=25, load_from_version=None, new_version_name='v0'):
   
    attack_templates = EnrollTemplates('attack_member', model, transform, embedding_agg)
    attack_templates_nonmember = EnrollTemplates('attack_nonmember', model, transform, embedding_agg)

    if load_from_version is not None:
        attack_templates.load(load_from_version)
        attack_templates_nonmember.load(load_from_version)
    else: 
        attack_data = member_data.sample(n_samples, random_state=random_state).reset_index(drop=True)
        attack_templates.enroll_templates(attack_data)
        attack_templates.save(new_version_name)
        
        attack_data_nomember = nonmember_data.sample(n_samples, random_state=random_state).reset_index(drop=True)
        attack_templates_nonmember.enroll_templates(attack_data_nomember)
        attack_templates_nonmember.save(new_version_name)

    return attack_templates, attack_templates_nonmember

def prepare_attack_searches(attack_templates, attack_templates_nonmember, similarity_threshold):
    search_attacks_member = [SearchTemplatePrivacyAttack(attack_template, attack_similarity_threshold=similarity_threshold) for _, attack_template in attack_templates.get_templates(as_pairs=True)]
    search_attacks_nonmember = [SearchTemplatePrivacyAttack(attack_template, attack_similarity_threshold=similarity_threshold) for _, attack_template in attack_templates_nonmember.get_templates(as_pairs=True)]

    return search_attacks_member, search_attacks_nonmember

def run_attack(benchmark, model, attack_templates, attack_templates_nonmember,
                T_match, T_accuracy=0.5, fpr_stat_threshold=0.05, load_from_version='v0'):
    search_attacks1, search_attacks0 = prepare_attack_searches(attack_templates, attack_templates_nonmember, T_match)

    res0 = []
    for a in search_attacks0:
        benchmark.run_benchmark(model, a, load_from_version=load_from_version, verbose=True)
        fnr1 = benchmark.get_fnr_at_fpr_top1(fpr_stat_threshold)
        res0.append(fnr1)

    res1 = []
    for a in search_attacks1:
        benchmark.run_benchmark(model, a, load_from_version=load_from_version, verbose=True)
        fnr1 = benchmark.get_fnr_at_fpr_top1(fpr_stat_threshold)
        res1.append(fnr1)
    

    out0 = np.array(res0) >= T_accuracy
    out1 = np.array(res1) >= T_accuracy

    fpr = out0.mean()
    tpr = out1.mean()

    return fpr, tpr, res0, res1