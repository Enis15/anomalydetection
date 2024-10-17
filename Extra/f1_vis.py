

def anomaly_vis(dataset_name):
    if dataset_name == 'dataset1':
        # Drop irrelavant features
        df = df.drop(['Unnamed: 0', 'trans_date_trans_time', 'trans_num', 'unix_time', 'dob', 'first', 'last', 'merch_zipcode'], axis=1)
        # Encoding categorical features with numerical variables
        cat_features = df.select_dtypes(include=['object']).columns
        for col in cat_features:
            df[col] = df[col].astype('category')
        df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

        features = df.drop(columns=['is_fraud'])


        # Preprocessing for dataset2
        if dataset_name == 'dataset2':
            # Drop specific features that aren't needed for correlation analysis
            # df = df.drop(['nameOrig', 'nameDest'], axis=1)
            # Map feature 'type' to numerical values
            df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})
            cat_features = df.select_dtypes(include=['object']).columns
            for col in cat_features:
                df[col] = df[col].astype('category')
            df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

            features = df.drop(columns=['isFraud'])

        # Preprocessing for dataset3
        if dataset_name == 'dataset3':
            # Drop specific features that aren't needed for correlation analysis
            df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)
            # Identify categorical values
            cat_features = df.select_dtypes(include=['object']).columns
            # Convert categorical features to numerical values
            for col in cat_features:
                df[col] = df[col].astype('category')  # Convert to category type
            df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

            features = df.drop(columns=['Fraud'])

        # Preprocessing for dataset4
        if dataset_name == 'dataset4':
            df['anomaly'] = df['anomaly'].replace({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})
            df['anomaly'] = df['anomaly'].astype(int)
            # Identify categorical values
            cat_features = df.select_dtypes(include=['object']).columns
            # Convert categorical features to numerical values
            for col in cat_features:
                df[col] = df[col].astype('category')  # Convert to category type
            df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)
            features = df.drop(columns=['anomaly'])

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features)

        df['TSNE1'] = tsne_results[:, 0]
        df['TSNE2'] = tsne_results[:, 1]

        plt.figure(figsize(10, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='anomaly', data=df, palette={0: 'blue', 1: 'red'})
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
        plt.title('t-SNE Plot for Dataset 4: Normal & Anomaly points')
        plt.savefig('../results/normal_anomaly{dataset_name}.png})')
        plt.show()