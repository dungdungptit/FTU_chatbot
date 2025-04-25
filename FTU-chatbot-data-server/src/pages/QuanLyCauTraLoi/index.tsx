/* eslint-disable radix */
/* eslint-disable no-underscore-dangle */
import TableBase from '@/components/Table';
import type { IAnwserRecord } from '@/models/answer';
import { IColumn } from '@/utils/interfaces';
import { DeleteOutlined, EditOutlined, EyeOutlined } from '@ant-design/icons';
import { Button, Card, Divider, Popconfirm, Image } from 'antd';
import React from 'react';
import { useModel, history } from 'umi';
import FormAnwser from './FormCauTraLoi';
import { ip } from '@/utils/ip';

const Index = () => {
  const answerModel = useModel('answer');

  const pathname = window.location.pathname;
  const IntentId = pathname.split('/')[2];

  const handleEdit = (record: IAnwserRecord) => {
    answerModel.setVisibleForm(true);
    answerModel.setEdit(true);
    answerModel.setRecord(record);
  };

  const handleDel = async (record: IAnwserRecord) => {
    if (record?.type_answer === 'image') {
      await answerModel.delWithImage(record?.id_answer ?? '').then(() => {
        answerModel.getData({ intent_id: IntentId });
      });
    } else {
      await answerModel.del(record?.id_answer ?? '').then(() => {
        answerModel.getData({ intent_id: IntentId });
      });
    }
  };

  const renderLast = (value: any, record: IAnwserRecord) => (
    <React.Fragment>
      {/* <Button
        type="primary"
        shape="circle"
        icon={<EyeOutlined />}
        title="Xem chi tiết"
        onClick={() => {
          console.log(record, 'record');
          answerModel.setRecord(record);
          history.push(`/answer/${record.id}`);
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
  const columns: IColumn<IAnwserRecord>[] = [
    {
      title: 'STT',
      dataIndex: 'index',
      width: 80,
      align: 'center',
    },
    {
      title: 'Nguồn thông tin',
      dataIndex: 'type_answer_choice',
      width: 200,
      align: 'left',
      render: (value: any, record: any) => {
        return (
          <Card title={value} bordered={false} style={{ width: 'max-content', padding: 0 }}>
            {record?.type_answer === 'text' ? (
              <p>{record?.answer}</p>
            ) : (
              <Image
                width={200}
                height={200}
                src={`${ip}${record?.image}`}
                style={{ objectFit: 'contain' }}
              />
            )}
          </Card>
        );
      },
    },
    {
      title: 'Thao tác',
      align: 'center',
      render: (value: any, record: IAnwserRecord) => renderLast(value, record),
      fixed: 'right',
      width: 80,
    },
  ];

  return (
    <>
      <TableBase
        modelName={'answer'}
        title="Nguồn thông tin"
        columns={columns}
        hascreate={true}
        formType={'Modal'}
        dependencies={[IntentId, answerModel.page, answerModel.limit, answerModel.condition]}
        widthDrawer={800}
        getData={answerModel.getData}
        Form={FormAnwser}
        noCleanUp={true}
        params={{
          intent_id: IntentId,
          page: answerModel.page,
          size: answerModel.limit,
          condition: answerModel.condition,
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
