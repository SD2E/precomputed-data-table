import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def get_strain_growth_plot(df, dpi=100):
    fig = plt.figure(dpi=dpi)
    ax = plt.axes()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
    columns = df.columns[1:] #[x for x in df.columns if 'strain' not in x]

    for i, row in df.iterrows():
        data = row[columns]
        ax.plot(row.keys()[1:].astype(float),data, label = row['strain'])
        ax.scatter(row.keys()[1:].astype(float),data, label=None)
    plt.ylabel("Post OD")
    plt.xscale("log", nonposx='clip')
    plt.xlabel("Inoculation OD")
    plt.title("Strain Post OD by Innoculation OD")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return ax

def get_per_experiment_od_plot(experiment_groups_result, pre='pre_OD600', post='OD600'):
    ax=plt.axes()
    ax.clear()
    ax.errorbar(experiment_groups_result[pre]['mean'], 
                experiment_groups_result[post]['mean'], 
                experiment_groups_result[pre,'std'], 
                experiment_groups_result[post,'std'],  
                alpha=.7,
                linestyle='None')
    ax.set_xlabel('Pre OD')
    ax.set_ylabel('Post OD')
    ax.set_ylim(0,5)
    ax.set_xlim(0,5)
    ax.set_aspect('equal', 'box')
    ax.set_title('Pre/Post OD by Experiment' )
    return ax

def get_per_experiment_od_by_od_plot(experiment_od_groups_result, pre='pre_OD600', post='OD600'):
    ods=experiment_od_groups_result.od.unique()
    f, axarr = plt.subplots(len(ods),1,                         
                        figsize=(60, 30))
    
    #f.subplots_adjust(wspace=2)
    for j, od in enumerate(ods):
        #my_df = df.loc[(df['strain_circuit'] == circuit] #.loc[result['strain'] == 'UWBF_NAND_01']
        m_my_df = experiment_od_groups_result.loc[(experiment_od_groups_result['od'] == od)] 


        axarr[j].errorbar(m_my_df[pre]['mean'], m_my_df[post]['mean'], m_my_df[pre,'std'], m_my_df[post,'std'],  alpha=.7,linestyle='None')

        axarr[j].set_xlabel('Pre OD')
        axarr[j].set_ylabel('Post OD')
        #axarr[j, i].set_xscale("log", nonposx='clip')
        #axarr[j, i].set_yscale("log", nonposy='clip')
        axarr[j].set_ylim(0,5)
        axarr[j].set_xlim(0,5)
        axarr[j].set_aspect('equal', 'box')
        axarr[j].set_title('Pre/Post OD per Experiment, OD = ' + str(od)  )
        #axarr[j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return f

def get_strain_statistics_by_od_plot(result, pre='pre_OD600', post='OD600', label_col='strain'):
    circuits = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
    ods = result.od.unique()
    ods.sort()
    f, axarr = plt.subplots(len(ods),6,                         
                        figsize=(60, 30))
    for j, od in enumerate(ods):
        for i, circuit in enumerate(circuits):
            m_my_df = result.loc[(result['strain_circuit'] == circuit) & \
                                 (result['od'] == od)] 

            colored_col = 'strain'
            colored_vals = np.sort(m_my_df[colored_col].dropna().unique())
            colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
            colordict = dict(zip(colored_vals, colors))  
            m_my_df.loc[:,"Color"] = m_my_df[colored_col].apply(lambda x: colordict[x])

            for strain in np.sort(m_my_df.strain.unique()):
                m_strain_df = m_my_df.loc[m_my_df['strain'] == strain]
                axarr[j, i].errorbar(m_strain_df[pre]['mean'],
                                     m_strain_df[post]['mean'],
                                     m_strain_df[pre,'std'],
                                     m_strain_df[post,'std'],  
                                     alpha=.7, 
                                     label=m_strain_df[label_col].values)

            axarr[j, i].set_xlabel('Pre OD')
            axarr[j, i].set_ylabel('Post OD')
            #axarr[j, i].set_xscale("log", nonposx='clip')
            #axarr[j, i].set_yscale("log", nonposy='clip')
            axarr[j, i].set_ylim(0,6)
            axarr[j, i].set_xlim(0,6)
            axarr[j, i].set_aspect('equal', 'box')
            axarr[j, i].set_title('OD = ' + str(od) + " " + circuit )
            axarr[j, i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return f

def get_strain_by_od_plot(df, result):

    circuits = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
    f, axarr = plt.subplots(1,6,                         
                        figsize=(100, 10))
    f.subplots_adjust(hspace=0.45, wspace=.25)
    for i, circuit in enumerate(circuits):
        my_df = df.loc[df['strain_circuit'] == circuit] #.loc[result['strain'] == 'UWBF_NAND_01']
        m_my_df = result.loc[result['strain_circuit'] == circuit] #.loc[result['strain'] == 'UWBF_NAND_01']


        colored_col = 'strain'
        colored_vals = np.sort(my_df[colored_col].dropna().unique())
        colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
        colordict = dict(zip(colored_vals, colors))  

        my_df.loc[:,"Color"] = my_df[colored_col].apply(lambda x: colordict[x])
        m_my_df.loc[:,"Color"] = m_my_df[colored_col].apply(lambda x: colordict[x])




        for strain in np.sort(my_df.strain.unique()):
            strain_df = my_df.loc[my_df['strain'] == strain]
            m_strain_df = m_my_df.loc[m_my_df['strain'] == strain]
            axarr[i].scatter(strain_df['OD600'], strain_df['od'], label=strain, color=strain_df.Color)
            axarr[i].scatter(m_strain_df['OD600']['mean'], 
                             m_strain_df['od'], 
                             s=m_strain_df['OD600']['std']*10000, 
                             alpha=.7, 
                             label=None, 
                             color=m_strain_df.Color)

        axarr[i].set_xlabel('Culture OD')
        axarr[i].set_ylabel('Innoculation OD')
        axarr[i].set_xscale("log", nonposx='clip')
        axarr[i].set_yscale("log", nonposy='clip')
        axarr[i].set_ylim(.00001, .1)
        axarr[i].set_title('Mean Corrected OD by Strain, ' + circuit)
        axarr[i].legend()
        
    return f
        
def get_pre_post_od_by_target_od(df, pre='pre_OD600', post='OD600', color='od'):
    ax = plt.axes()
    fig_size = plt.rcParams["figure.figsize"]
    # Set figure width to 12 and height to 9
    fig_size[0] = 40
    fig_size[1] = 30
    plt.rcParams["figure.figsize"] = fig_size
    colored_vals = np.sort(df[color].dropna().unique())
    colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
    colordict = dict(zip(colored_vals, colors))  

    df["Color"] = df[color].apply(lambda x: colordict[x])

    ods = df['od'].astype(float).unique()
    ods.sort()
    for od in ods:
        ax.scatter(df[pre], df[post], label=str(od), c=df['Color'], alpha=0.5)
    plt.ylabel("Post OD")
    plt.xlabel("Pre OD")
    plt.title("Pre vs Post OD by Inoculation OD")
    plt.legend()
    legend = ax.get_legend()
    if color == 'od':
        for i, od in enumerate(ods):
            legend.legendHandles[i].set_color(colordict[od])

    return ax


def get_od_post_od_scatter(df, od='od', post='post_od_raw', color='part_2_id'):
    ax = plt.axes()
    fig_size = plt.rcParams["figure.figsize"]
    # Set figure width to 12 and height to 9
    fig_size[0] = 16
    fig_size[1] = 12

    plt.rcParams["figure.figsize"] = fig_size
    colored_vals = np.sort(df[color].astype(str).dropna().unique())
    colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
    colordict = dict(zip(colored_vals, colors))  

    df["Color"] = df[color].apply(lambda x: colordict[str(x)])

    exp = df[color].astype(str).unique()
    exp.sort()
    for ex in exp:
        ax.scatter(df[od], df[post], label=str(ex), c=df['Color'], alpha=0.5)
    plt.ylabel("Post OD")
    plt.xlabel("Inoculation OD")
    plt.title("Inoculation OD vs Post OD")
    plt.legend()
    legend = ax.get_legend()
#    if color == 'part_2_id':
    for i, ex in enumerate(exp):
        legend.legendHandles[i].set_color(colordict[str(ex)])

    return ax



def get_strain_inoculation_to_final_od(df, pre='pre_OD600', post='OD600', color='od'):
    ax = plt.axes()
    #fig_size = plt.rcParams["figure.figsize"]
    # Set figure width to 12 and height to 9
    #fig_size[0] = 4
    #fig_size[1] = 3
    #plt.rcParams["figure.figsize"] = fig_size
    colored_vals = np.sort(df[color].dropna().unique())
    colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
    colordict = dict(zip(colored_vals, colors))  

    df["Color"] = df[color].apply(lambda x: colordict[x])

    ods = df['od'].unique()
    for od in ods:
        ax.scatter(df['od'], df[post],  label = df['strain'], alpha=0.5)
    plt.ylabel("Post OD")
    plt.xlabel("Inoculation OD")
    plt.title("Pre vs Post OD by Innoculation OD")
    plt.legend()
    legend = ax.get_legend()
    ax.set_ylim(0,6)
    ax.set_xlim(0,6)


    if color == 'od':
        for i, od in enumerate(ods):
            legend.legendHandles[i].set_color(colordict[od])

    return ax




