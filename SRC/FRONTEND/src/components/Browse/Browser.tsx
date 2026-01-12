import React, { useMemo, useState } from 'react';
import dataset from '../../data/dataset.json';

interface RecordType {
    id: string;
    diagnosis: string;
    symptoms?: string;
    treatment?: string;
}

export const BrowseRecords: React.FC = () => {
    const records: RecordType[] = dataset as RecordType[];
    const [query, setQuery] = useState<string>('');
    const [page, setPage] = useState<number>(1);
    const pageSize = 5;
    const [expandedId, setExpandedId] = useState<string | null>(null);

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!q) return records;
        return records.filter(r =>
            r.diagnosis.toLowerCase().includes(q) ||
            (r.symptoms || '').toLowerCase().includes(q) ||
            (r.treatment || '').toLowerCase().includes(q)
        );
    }, [records, query]);

    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    const pageItems = filtered.slice((page - 1) * pageSize, page * pageSize);

    function toggleExpand(id: string) {
        setExpandedId(prev => (prev === id ? null : id));
    }

    return (
        <div className="browse-container">
            <div className="search-wrap">
                <input
                    className="search-input"
                    aria-label="Search records"
                    placeholder="Search by diagnosis, symptoms, or treatment"
                    value={query}
                    onChange={e => {
                        setQuery(e.target.value);
                        setPage(1);
                    }}
                />
            </div>

            <div className="records-list">
                <table className="records-table">
                    <thead>
                        <tr>
                            <th style={{ textAlign: 'left' }}>Diagnosis</th>
                            <th style={{ textAlign: 'left' }}>Symptoms</th>
                            <th style={{ textAlign: 'left' }}>Treatment</th>
                            <th style={{ width: 120 }}>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {pageItems.map(rec => (
                            <React.Fragment key={rec.id}>
                                <tr className="record-row">
                                    <td>
                                        <div className="diagnosis">{rec.diagnosis}</div>
                                        <div className="record-id">{rec.id}</div>
                                    </td>
                                    <td className="cell-snippet">{rec.symptoms ? (rec.symptoms.length > 120 ? rec.symptoms.slice(0, 120) + '…' : rec.symptoms) : ''}</td>
                                    <td className="cell-snippet">{rec.treatment ? (rec.treatment.length > 120 ? rec.treatment.slice(0, 120) + '…' : rec.treatment) : ''}</td>
                                    <td>
                                        <button className="btn" onClick={() => toggleExpand(rec.id)}>
                                            {expandedId === rec.id ? 'Hide' : 'Details'}
                                        </button>
                                    </td>
                                </tr>
                                {expandedId === rec.id && (
                                    <tr className="expanded-row">
                                        <td colSpan={4} className="details">
                                            {rec.symptoms && (
                                                <p>
                                                    <strong>Symptoms:</strong> {rec.symptoms}
                                                </p>
                                            )}
                                            {rec.treatment && (
                                                <p>
                                                    <strong>Treatment:</strong> {rec.treatment}
                                                </p>
                                            )}
                                        </td>
                                    </tr>
                                )}
                            </React.Fragment>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="pagination">
                <div className="page-info">Page {page} of {totalPages}</div>
                <div>
                    <button className="btn" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1}>
                        Previous
                    </button>
                    <button className="btn" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages}>
                        Next
                    </button>
                </div>
            </div>
        </div>
    );
};