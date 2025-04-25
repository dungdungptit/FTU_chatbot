/* eslint-disable radix */
/* eslint-disable no-underscore-dangle */
import TableBase from '@/components/Table';
import type { IQuestionRecord } from '@/models/question';
import { IColumn } from '@/utils/interfaces';
import { DeleteOutlined, EditOutlined, EyeOutlined } from '@ant-design/icons';
import { Button, Card, Divider, Popconfirm } from 'antd';
import React from 'react';
import { useModel, history } from 'umi';
import FormQuestion from './FormCauHoi';

const Index = () => {
  const questionModel = useModel('question');

  const pathname = window.location.pathname;
  const IntentId = pathname.split('/')[4];

  const handleEdit = (record: IQuestionRecord) => {
    questionModel.setVisibleForm(true);
    questionModel.setEdit(true);
    questionModel.setRecord(record);
  };

  const handleDel = async (record: IQuestionRecord) => {
    await questionModel.del(record?.id ?? '').then(() => {
      questionModel.getData({ intent_id: IntentId });
    });
  };

  const renderLast = (value: any, record: IQuestionRecord) => (
    <React.Fragment>
      {/* <Button
        type="primary"
        shape="circle"
        icon={<EyeOutlined />}
        title="Xem chi tiết"
        onClick={() => {
          console.log(record, 'record');
          questionModel.setRecord(record);
          history.push(`/question/${record.id}`);
        }}
      /> */}
      {/* <Divider type="vertical" /> */}
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
  const columns: IColumn<IQuestionRecord>[] = [
    {
      title: 'STT',
      dataIndex: 'index',
      width: 80,
      align: 'center',
    },
    {
      title: 'Câu hỏi',
      dataIndex: 'question',
      width: 200,
      align: 'left',
    },
    {
      title: 'Câu trả lời',
      dataIndex: 'answer',
      render: (value: any, record: IQuestionRecord) => (<p>{value.slice(0, value.length > 150 ? 150 : value.length)}{value.length > 150 && "..."}</p>),
      width: 200,
      align: 'left',
    },
    {
      title: 'Thao tác',
      align: 'center',
      render: (value: any, record: IQuestionRecord) => renderLast(value, record),
      fixed: 'right',
      width: 80,
    },
  ];

  return (
    <>
      <TableBase
        modelName={'question'}
        title="Danh sách câu hỏi"
        columns={columns}
        hascreate={true}
        formType={'Modal'}
        dependencies={[IntentId, questionModel.page, questionModel.limit, questionModel.condition]}
        widthDrawer={800}
        getData={() => questionModel.getData({ intent_id: IntentId })}
        Form={FormQuestion}
        noCleanUp={true}
        params={{
          "filters[intent][$eq]": IntentId,
          "pagination[page]": questionModel.page,
          "pagination[pageSize]": questionModel.limit,
          condition: questionModel.condition,
        }}
        maskCloseableForm={true}
        otherProps={{
          scroll: {
            x: 1000,
          },
        }}
      />
    </>
  );
};

export default Index;
