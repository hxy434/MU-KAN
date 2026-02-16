import React from 'react';
import { Tabs, Image, Card } from 'antd';
import { useTranslation } from 'react-i18next';

export default function ResultTabs({ results }) {
  const { t } = useTranslation();
  return (
    <Card style={{ marginTop: 24, borderRadius: 12, boxShadow: '0 2px 12px #eee' }}>
      <Tabs tabBarStyle={{ fontSize: 18, fontWeight: 500 }}>
        <Tabs.TabPane tab={t('original')} key="original">
          <Image src={results.original} width={400} style={{ borderRadius: 8, boxShadow: '0 2px 8px #ccc' }} />
        </Tabs.TabPane>
        <Tabs.TabPane tab={t('mask')} key="mask">
          <Image src={results.mask} width={400} style={{ borderRadius: 8, boxShadow: '0 2px 8px #ccc' }} />
        </Tabs.TabPane>
        <Tabs.TabPane tab={t('overlay')} key="overlay">
          <Image src={results.overlay} width={400} style={{ borderRadius: 8, boxShadow: '0 2px 8px #ccc' }} />
        </Tabs.TabPane>
        <Tabs.TabPane tab={t('contour')} key="contour">
          <Image src={results.contour} width={400} style={{ borderRadius: 8, boxShadow: '0 2px 8px #ccc' }} />
        </Tabs.TabPane>
        <Tabs.TabPane tab={t('stats')} key="stats">
          <Image src={results.stats} width={400} style={{ borderRadius: 8, boxShadow: '0 2px 8px #ccc' }} />
        </Tabs.TabPane>
        <Tabs.TabPane tab={t('report')} key="report">
          <Card style={{ background: '#f6f8fa', borderRadius: 8 }}>
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: 16 }}>{results.report}</pre>
          </Card>
        </Tabs.TabPane>
      </Tabs>
    </Card>
  );
} 