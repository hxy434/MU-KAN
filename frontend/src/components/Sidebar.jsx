import React, { useState } from 'react';
import { Input, Button, Divider } from 'antd';
import { PlusOutlined, DeleteOutlined, SearchOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

function groupByModelAndDate(history) {
  const groups = {};
  history.forEach(item => {
    const model = item.model || 'Other';
    const latestRound = item.rounds && item.rounds.length > 0 ? item.rounds[item.rounds.length - 1] : null;
    const latestTime = latestRound ? (latestRound.user?.time || latestRound.ai?.time) : item.time;
    const date = latestTime ? latestTime.split(' ')[0] : 'Unknown';
    if (!groups[model]) groups[model] = {};
    if (!groups[model][date]) groups[model][date] = [];
    groups[model][date].push(item);
  });
  return groups;
}

export default function Sidebar({ history, activeId, onSelect, onNew, onClear }) {
  const { t } = useTranslation();
  const [search, setSearch] = useState('');
  const groups = groupByModelAndDate(
    search
      ? history.filter(h => h.filename?.toLowerCase().includes(search.toLowerCase()) || h.model?.toLowerCase().includes(search.toLowerCase()) || h.time?.includes(search))
      : history
  );
  return (
    <div className="sidebar">
      <div style={{ display: 'flex', alignItems: 'center', padding: '18px 20px 8px 20px', gap: 8 }}>
        <Button type="primary" icon={<PlusOutlined />} shape="round" onClick={onNew} style={{ fontWeight: 600 }}>{t('new_analysis')}</Button>
      </div>
      <Input
        className="sidebar-search"
        placeholder={t('search')}
        prefix={<SearchOutlined />}
        allowClear
        value={search}
        onChange={e => setSearch(e.target.value)}
        style={{ margin: '0 0 0 0', borderRadius: 0 }}
      />
      <Divider style={{ margin: '10px 0' }} />
      {Object.keys(groups).length === 0 && (
        <div className="empty-state">{t('no_history')}</div>
      )}
      {Object.entries(groups).map(([model, dateGroup]) => (
        <div className="sidebar-group" key={model}>
          <div className="sidebar-title"><span className="sidebar-label">{model}</span></div>
          {Object.entries(dateGroup).map(([date, items]) => (
            <div key={date}>
              <div className="sidebar-meta" style={{ margin: '8px 0 4px 24px', color: '#1976d2', fontWeight: 500 }}>
                <span style={{ marginRight: 6 }}>📅</span>{date}
              </div>
              {items.map(record => (
                <div
                  className={"sidebar-history-item" + (activeId === record.id ? ' active' : '')}
                  key={record.id}
                  onClick={() => onSelect(record.id)}
                >
                  <img src={record.preview} alt="缩略图" className="sidebar-thumb" />
                  <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 2 }}>
                    {record.rounds && record.rounds.length > 0 
                      ? record.rounds[record.rounds.length - 1].user.filename 
                      : record.filename || t('unknown_file')}
                  </div>
                  <div className="sidebar-meta">{(record.rounds && record.rounds.length > 0 ? (record.rounds[record.rounds.length - 1].user?.time || record.rounds[record.rounds.length - 1].ai?.time) : record.time).split(' ')[1]}</div>
                </div>
              ))}
            </div>
          ))}
        </div>
      ))}
      <div style={{ flex: 1 }} />
      <div style={{ padding: 16, borderTop: '1px solid #f0f0f0' }}>
        <Button icon={<DeleteOutlined />} danger block onClick={onClear}>{t('clear_history')}</Button>
      </div>
    </div>
  );
} 
