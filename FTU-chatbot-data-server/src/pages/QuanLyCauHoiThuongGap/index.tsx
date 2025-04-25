/* eslint-disable radix */
/* eslint-disable no-underscore-dangle */
import TableBase from '@/components/Table';
import type { IF, IFqasRecord, IFqasRecordqasRecord } from '@/models/fqas';
import { IColumn } from '@/utils/interfaces';
import { DeleteOutlined, DislikeOutlined, EditOutlined, EyeOutlined, LikeOutlined } from '@ant-design/icons';
import { Button, Card, Divider, Popconfirm } from 'antd';
import React from 'react';
import { useModel, history } from 'umi';
import FormQuestion from './FormCauHoi';

const Index = () => {
  const fqasModel = useModel('fqas');

  const pathname = window.location.pathname;
  const IntentId = pathname.split('/')[4];

  const handleEdit = (record: IFqasRecord) => {
    console.log(record);
    fqasModel.setRecord(record);
    fqasModel.setVisibleForm(true);
    fqasModel.setEdit(true);
  };

  const handleDel = async (record: IFqasRecord) => {
    await fqasModel.del(record?.id ?? '').then(() => {
      fqasModel.getData({ intent_id: IntentId });
    });
  };

  const renderLast = (value: any, record: IFqasRecord) => (
    <React.Fragment>
      {/* <Button
        type="primary"
        shape="circle"
        icon={<EyeOutlined />}
        title="Xem chi tiết"
        onClick={() => {
          console.log(record, 'record');
          fqasModel.setRecord(record);
          history.push(`/fqas/${record.id}`);
        }}
      />
      <Divider type="vertical" /> */}
      <Button
        type="primary"
        shape="circle"
        icon={<EditOutlined />}
        title="Chỉnh sửa"
        onClick={() => {
          console.log(record, 'record');
          handleEdit(record);
        }}
      />
      <Divider type="vertical" />
      <Popconfirm
        title="Bạn có muốn xóa?"
        okText="Có"
        cancelText="Không"
        onConfirm={() => handleDel(record)}
      >
        <Button type="danger" shape="circle" icon={<DeleteOutlined />} title="Xóa" />
      </Popconfirm>
    </React.Fragment>
  );
  const columns: IColumn<IFqasRecord>[] = [
    {
      title: 'STT',
      dataIndex: 'index',
      width: 80,
      align: 'center',
    },
    {
      title: 'Câu hỏi',
      dataIndex: 'question',
      width: 120,
      align: 'left',
    },
    {
      title: 'Chatbot trả lời',
      dataIndex: 'chatbot_answer',
      width: 300,
      align: 'left',
      render: (value: any, record: IFqasRecord) => (<p>{value.slice(0, value.length > 150 ? 150 : value.length)}{value.length > 150 && "..."}</p>),
    },
    {
      title: 'Ngường dùng phản hồi',
      dataIndex: 'human_answer',
      width: 300,
      align: 'left',
      render: (value: any, record: IFqasRecord) => (<p>{value.slice(0, value.length > 150 ? 150 : value.length)}{value.length > 150 && "..."}</p>),
    },
    {
      title: 'Đánh giá',
      dataIndex: 'like',
      width: 80,
      align: 'center',
      render: (value: any, record: IFqasRecord) => (value == true ? <LikeOutlined style={{ color: "green" }} /> : <DislikeOutlined style={{ color: "red" }} />),
    },
    {
      title: 'Duyệt',
      dataIndex: 'used',
      width: 80,
      align: 'center',
      render: (value: any, record: IFqasRecord) => (value == true ? "Duyệt" : "Không duyệt"),
    },
    {
      title: 'Thao tác',
      align: 'center',
      render: (value: any, record: IFqasRecord) => renderLast(value, record),
      fixed: 'right',
      width: 200,
    },
  ];

  return (
    <>
      <TableBase
        modelName={'fqas'}
        title="Danh sách câu hỏi"
        columns={columns}
        hascreate={false}
        formType={'Modal'}
        dependencies={[fqasModel.page, fqasModel.limit, fqasModel.condition]}
        widthDrawer={800}
        onRow={(record, rowIndex) => {
          return {
            onClick: (event) => {
              // handleEdit(record)
            }, // click row
          };
        }}
        getData={() => fqasModel.getData({ intent_id: IntentId })}
        Form={FormQuestion}
        noCleanUp={true}
        params={{
          "pagination[page]": fqasModel.page,
          "pagination[pageSize]": fqasModel.limit,
          condition: fqasModel.condition,
        }}
        maskCloseableForm={true}
        otherProps={{
          scroll: {
            x: 1000,
            // y: 240
          },
        }}
      />
    </>
  );
};

export default Index;
