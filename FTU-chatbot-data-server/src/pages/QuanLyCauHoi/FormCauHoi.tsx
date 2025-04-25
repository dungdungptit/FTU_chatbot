/* eslint-disable no-underscore-dangle */
import { IQuestionRecord } from '@/models/question';
import { ip } from '@/utils/ip';
import rules from '@/utils/rules';
import {
  Form,
  Input,
  Button,
  Card,
  Spin,
  DatePicker,
  InputNumber,
  Col,
  Row,
  Select,
  Tabs,
  Divider,
} from 'antd';
import TextArea from 'antd/lib/input/TextArea';
import React, { useEffect, useState } from 'react';
import { useModel } from 'umi';

const layout = {
  labelCol: { span: 6 },
  wrapperCol: { span: 16 },
};
const tailLayout = {
  wrapperCol: { offset: 8, span: 16 },
};

const { TabPane } = Tabs;

const FormQuestion = () => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<boolean>(false);
  const questionModel = useModel('question');
  const pathname = window.location.pathname;
  const IntentId = pathname.split('/')[4];

  const handleFinish = async (values: IQuestionRecord) => {
    if (questionModel.edit) {
      if (questionModel?.record?.id) {
        const payload = {
          data: {
            question: values.question,
            answer: values.answer,
            intent: {
              id: IntentId,
            },
          }
        }
        await questionModel.upd({ id: questionModel?.record?.id, data: payload });
        questionModel.setVisibleForm(false);
        questionModel.setEdit(false);
        questionModel.getDataById(questionModel?.record?.id ?? '');
        questionModel.getData({ intent_id: IntentId });
      } else {
        const payload = {
          data: {
            question: values.question,
            answer: values.answer,
            intent: {
              id: IntentId,
            },
          }
        }
        await questionModel.add(payload);
        questionModel.setVisibleForm(false);
        questionModel.setEdit(false);
        questionModel.getData({ intent_id: IntentId });
      }
    } else {
      const payload = {
        data: {
          question: values.question,
          answer: values.answer,
          intent: {
            id: IntentId,
          },
        }
      }
      await questionModel.add(payload);
      questionModel.setVisibleForm(false);
      questionModel.setEdit(false);
      questionModel.getData({ intent_id: IntentId });
    }
  };

  const handleFinishFile = async (values: any) => {
    const data = document.getElementById('formQuestionFile') as HTMLFormElement;
    const formData = new FormData(data);
    formData.append('id', IntentId);
    await questionModel.addFileQuestion(formData);
    questionModel.setVisibleForm(false);
    questionModel.setEdit(false);
    questionModel.getData({ intent_id: IntentId });
  };

  console.log(form.getFieldsValue(), 'form.getFieldValue()');
  return (
    <Spin spinning={questionModel.loading}>
      {questionModel.edit ? (
        <Card title={'Chỉnh sửa'}>
          <Form
            id="formQuestion"
            // {...layout}
            layout='vertical'
            form={form}
            onFinish={handleFinish}
            initialValues={{
              ...(questionModel?.record ?? {}),
            }}
          >
            <Form.Item label="Mã câu hỏi" name="question_id" hidden={true}>
              <Input
                placeholder="Mã câu hỏi"
                name="question_id"
                defaultValue={questionModel?.record?.id ?? ''}
                disabled
              />
            </Form.Item>
            <Form.Item label="Câu hỏi" name="question">
              <Input placeholder="Nhập câu hỏi" name="question" />
            </Form.Item>
            <Form.Item label="Câu trả lời" name="answer">
              <TextArea placeholder="Nhập câu trả lời" name="answer" style={{
                minHeight: '200px'
              }} />
            </Form.Item>
            <Form.Item {...tailLayout}>
              <Button type="primary" htmlType="submit">
                {questionModel.edit ? 'Cập nhật' : 'Thêm mới'}
              </Button>
            </Form.Item>
          </Form>
        </Card>
      ) : (
        <Card title={'Thêm mới'}>
          <Tabs>
            <TabPane tab="Nhập dữ liệu" key="tab-a">
              <Form
                id="formQuestion"
                // {...layout}
                layout='vertical'
                form={form}
                onFinish={handleFinish}
                initialValues={{
                  ...(questionModel?.record ?? {}),
                }}
              >
                <Form.Item label="Câu hỏi" name="question">
                  <Input placeholder="Nhập câu hỏi" name="question" />
                </Form.Item>
                <Form.Item label="Câu trả lời" name="answer">
                  <Input placeholder="Nhập câu trả lời" name="answer" />
                </Form.Item>
                <Form.Item {...tailLayout}>
                  <Button type="primary" htmlType="submit">
                    {'Thêm mới'}
                  </Button>
                </Form.Item>
              </Form>
            </TabPane>
            {/* <TabPane tab="Nhập bằng file" key="tab-b">
              <Form {...layout} form={form} onFinish={handleFinishFile} id="formQuestionFile">
                <Form.Item label="Chủ đề" name="question_id">
                  <Input
                    placeholder="Mã chủ đề"
                    name="question_id"
                    defaultValue={IntentId}
                    disabled
                  />
                </Form.Item>
                <Form.Item label="File" name="file">
                  <Input type="file" name="file" />
                </Form.Item>
                <Form.Item {...tailLayout}>
                  <Button type="primary" htmlType="submit">
                    Thêm mới
                  </Button>
                </Form.Item>
              </Form>
              <Button type="primary">
                <a
                  download={'tep_tin_mau_cau_hoi.xlsx'}
                  href={`${ip}/media/format/question_post/tep_tin_mau_cau_hoi.xlsx`}
                >
                  Tải file mẫu
                </a>
              </Button>
              <br />
              <Card title="Mẫu file">
                <Row>
                  <Col span={12}>
                    <strong>Câu hỏi</strong>
                  </Col>
                </Row>
                <Divider />
                <Row>
                  <Col span={12}>
                    <p>Câu hỏi thuộc chủ đề(có thể để trống)</p>
                  </Col>
                </Row>
              </Card>
            </TabPane> */}
          </Tabs>
        </Card>
      )}
    </Spin>
  );
};

export default FormQuestion;
